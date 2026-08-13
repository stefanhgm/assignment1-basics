"""Byte-level BPE tokenizer training (CS336 assignment 1).

Pipeline:
  1. Chunk the input file at <|endoftext|> boundaries (enables parallel work).
  2. Pretokenize each chunk in parallel (GPT-2 regex), count unique pretokens.
  3. Iteratively merge the most frequent adjacent token pair until vocab_size,
     maintaining pair counts and a pair -> words index incrementally.

Core data structures (inside train_bpe):
  vocab:                dict[int, bytes]            token id -> the bytes it spells
  merges:               list[tuple[bytes, bytes]]   merge rules, in creation order (order = rank)
  token_word_freq:      Counter[tuple[int, ...]]    pretoken (as token-id tuple) -> corpus frequency
  pair_counts:          Counter[tuple[int, int]]    adjacent token pair -> total corpus frequency
  pairs_to_token_words: dict[pair, set[word]]       inverted index: pair -> words containing it
"""

import os
from collections import Counter, defaultdict
from itertools import pairwise
from multiprocessing import Pool
from typing import BinaryIO

import regex as re  # third-party `regex`, needed for \p{L} / \p{N}

# GPT-2 pretokenization pattern. Matches (in order): contractions, ` word`,
# ` number`, ` punctuation`, trailing whitespace, other whitespace runs.
# Every input char lands in exactly one pretoken, so pretokens concatenate
# back to the original text.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find byte offsets that split the file into ~equal chunks, with each
    boundary aligned to an occurrence of `split_special_token` so that no
    chunk cuts a document in half. May return fewer chunks than requested.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Uniformly spaced initial guesses; each is then advanced to the next
    # occurrence of the special token.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # scan forward in 4 KiB steps

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":  # hit EOF: clamp boundary to file end
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Deduplicate: boundaries can collapse on small files.
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(path: str, start: int, end: int, special_tokens: list[str]) -> Counter:
    """Worker: read file[start:end], split out special tokens, pretokenize,
    and return Counter[tuple[int, ...] -> count] of UTF-8 byte tuples.

    Module-level (not nested) so it can be pickled by multiprocessing.
    """
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts = Counter()
    # Split on special tokens first so no merge can ever span a document
    # boundary (and the tokens themselves never enter the merge process).
    for part in re.split("|".join(re.escape(t) for t in special_tokens), chunk):
        counts.update(tuple(word.encode("utf-8")) for word in re.findall(PAT, part))
    return counts


def get_pairs(token_word_freq: Counter) -> Counter:
    """Full pair-count table: adjacent pair -> total frequency over the corpus.
    Called once to initialize; afterwards counts are updated incrementally.
    """
    pairs = Counter()
    for word, freq in token_word_freq.items():
        for pair in pairwise(word):
            pairs[pair] += freq
    return pairs


def create_pairs_to_token_words(token_word_freq: Counter) -> defaultdict:
    """Inverted index: pair -> set of words containing that pair.
    Lets a merge touch only the affected words instead of the whole corpus.
    """
    pairs_to_token_words = defaultdict(set)
    for token_word in token_word_freq:
        for pair in pairwise(token_word):
            pairs_to_token_words[pair].add(token_word)
    return pairs_to_token_words


def merge_word(pair: tuple, new_id: int, byte_word: tuple) -> tuple:
    """Replace every non-overlapping occurrence of `pair` in `byte_word`
    with `new_id` (left to right)."""
    new_bytes = []
    j = 0
    while j < len(byte_word):
        if j < len(byte_word) - 1 and (byte_word[j], byte_word[j + 1]) == pair:
            new_bytes.append(new_id)
            j += 2  # skip both merged elements
        else:
            new_bytes.append(byte_word[j])
            j += 1
    return tuple(new_bytes)


def merge_pairs(pair, new_id, token_word_freq, pairs_to_token_words, pair_counts):
    """Apply one merge. Only words containing `pair` are rewritten; pair
    counts and the inverted index are patched by retracting each old word's
    pairs and inserting the new word's pairs (weighted by word frequency).
    """
    # list(...) snapshot: the loop body discards from this very set.
    for byte_word in list(pairs_to_token_words[pair]):
        merged_byte_word = merge_word(pair, new_id, byte_word)
        freq = token_word_freq.pop(byte_word)
        token_word_freq[merged_byte_word] = freq

        # Retract the old word entirely...
        for p in pairwise(byte_word):
            pairs_to_token_words[p].discard(byte_word)
            pair_counts[p] -= freq
            if pair_counts[p] == 0:
                del pair_counts[p]  # keep dead pairs out of max()

        # ...then insert the new word. (Full retract/insert is O(word length)
        # and immune to overlapping-pair edge cases.)
        for p in pairwise(merged_byte_word):
            pairs_to_token_words[p].add(merged_byte_word)
            pair_counts[p] += freq

    return token_word_freq, pairs_to_token_words, pair_counts


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns:
      vocab:  dict[int, bytes] of size `vocab_size`
              (256 byte tokens + special tokens + learned merges)
      merges: list[tuple[bytes, bytes]] in the order they were learned
    """
    # Base vocab: all 256 single bytes, then special tokens.
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    merges = []

    # --- Pretokenization (parallel over special-token-aligned chunks) ---
    num_processes = os.cpu_count() or 4
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    args = [(input_path, s, e, special_tokens) for s, e in pairwise(boundaries)]
    with Pool(num_processes) as pool:
        results = pool.starmap(pretokenize_chunk, args)

    # Per-chunk counters sum exactly because merges never cross chunks.
    token_word_freq = Counter()
    for c in results:
        token_word_freq.update(c)

    # --- One-time initialization; both updated incrementally afterwards ---
    pairs_to_token_words = create_pairs_to_token_words(token_word_freq)
    pair_counts = get_pairs(token_word_freq)

    # --- Merge loop ---
    while len(vocab) < vocab_size:
        if not pair_counts:  # corpus fully merged before reaching vocab_size
            break

        # Most frequent pair; ties broken by lexicographically greatest
        # byte strings (NOT token ids -- id order != byte order once ids >= 256).
        most_freq_token_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        new_id = len(vocab)
        token_word_freq, pairs_to_token_words, pair_counts = merge_pairs(
            most_freq_token_pair, new_id, token_word_freq, pairs_to_token_words, pair_counts
        )

        vocab[new_id] = vocab[most_freq_token_pair[0]] + vocab[most_freq_token_pair[1]]
        merges.append((vocab[most_freq_token_pair[0]], vocab[most_freq_token_pair[1]]))

    return vocab, merges


if __name__ == "__main__":
    # Guard is required: multiprocessing workers re-import this module.
    tiny_stories = "data/TinyStoriesV2-GPT4-train.txt"

    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(tiny_stories, 280, special_tokens)
