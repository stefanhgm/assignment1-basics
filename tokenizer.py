import os
from collections import Counter, defaultdict
from itertools import pairwise
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(input_path, vocab_size, special_tokens, **kwargs):

    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    merges = []

    # Pretokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # PAT = r"""\S+"""

    def get_pairs(token_word_freq):
        pairs = Counter()
        for bw, c in token_word_freq.items():
            token_pairs = Counter()
            token_pairs.update([pair for pair in pairwise(bw)])
            pairs.update({pair: pair_count * c for pair, pair_count in token_pairs.items()})
            # print(word_pairs)
            # print(pairs)
        return pairs

    def merge_word(pair, id, byte_word):
        new_bytes = []
        j = 0
        while j < len(byte_word):
            if j < len(byte_word) - 1 and byte_word[j] == pair[0] and byte_word[j + 1] == pair[1]:
                new_bytes.append(id)
                j += 2
            else:
                new_bytes.append(byte_word[j])
                j += 1
        return tuple(new_bytes)

    def create_pairs_to_token_words(token_word_freq):
        pairs_to_token_words = defaultdict(set)
        for token_word in token_word_freq:
            for pair in pairwise(token_word):
                pairs_to_token_words[pair].add(token_word)
        return pairs_to_token_words

    def merge_pairs(pair, id, token_word_freq, pairs_to_token_words, pair_counts):
        for byte_word in list(pairs_to_token_words[pair]):
            merged_byte_word = merge_word(pair, id, byte_word)
            freq = token_word_freq.pop(byte_word)
            token_word_freq[merged_byte_word] = freq

            # update all relevant pairs to token mappings
            # Remove old indices
            for p in pairwise(byte_word):
                pairs_to_token_words[p].discard(byte_word)
                pair_counts[p] -= freq
                if pair_counts[p] == 0:
                    del pair_counts[0]

            for p in pairwise(merged_byte_word):
                pairs_to_token_words[p].add(merged_byte_word)
                pair_counts[p] += freq

        return token_word_freq, pairs_to_token_words, pair_counts

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # Keep words with their frequencies and transform into bytes
        token_word_freq = Counter()
        for start, end in pairwise(boundaries):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            for part in re.split("|".join(re.escape(t) for t in special_tokens), chunk):
                token_word_freq.update(tuple(word.encode("utf-8")) for word in re.findall(PAT, part))

        pairs_to_token_words = create_pairs_to_token_words(token_word_freq)
        pair_counts = get_pairs(token_word_freq)

        # pre_input = re.findall(PAT, chunk)
        # print(token_word_freq)
        # print()
        # print(pairs_to_token_words)

        # Transform into byte sequences
        # bytes_word_freq = {s.encode("utf-8"): c for s, c in word_freq.items()}
        # print(bytes_word_freq)

    # Fill vocab with most common pair until vocab size reached and update data structures
    while len(vocab) < vocab_size:
        most_freq_token_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))

        token_word_freq, pairs_to_token_words, pair_counts = merge_pairs(
            most_freq_token_pair, len(vocab), token_word_freq, pairs_to_token_words, pair_counts
        )

        # byte_pre_input = merge_pairs(best, len(vocab), byte_pre_input)
        vocab[len(vocab)] = vocab[most_freq_token_pair[0]] + vocab[most_freq_token_pair[1]]
        merges.append((vocab[most_freq_token_pair[0]], vocab[most_freq_token_pair[1]]))

    return vocab, merges


# Main program
minimal_example_path = "bpe_example.txt"
tiny_stories_50 = "data/TinyStoriesV2-GPT4-train_50.txt"
tiny_stories_50k = "data/TinyStoriesV2-GPT4-train_50k.txt"
tiny_stories_500k = "data/TinyStoriesV2-GPT4-train_500k.txt"
tiny_stories = "data/TinyStoriesV2-GPT4-train.txt"
tiny_stories_valid = "data/TinyStoriesV2-GPT4-valid.txt"

special_tokens = ["<|endoftext|>"]
vocab, merges = train_bpe(tiny_stories_500k, 280, special_tokens)
print(vocab, merges)
