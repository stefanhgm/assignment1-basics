import os
from typing import BinaryIO
import regex as re
from collections import Counter


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

    def get_pairs(list_bytes):
        pairs = Counter()
        pairs.update([pair for bytes in list_bytes for pair in zip(bytes, bytes[1:])])
        return pairs

    def merge_pairs(pair, id, list_bytes):
        result = []
        for bytes in list_bytes:
            new_bytes = []
            j = 0
            while j < len(bytes):
                if j < len(bytes) - 1 and bytes[j] == pair[0] and bytes[j+1] == pair[1]:
                    new_bytes.append(id)
                    j += 2
                else:
                    new_bytes.append(bytes[j])
                    j += 1
            result.append(new_bytes)

        return result
    
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        byte_pre_input = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pre_input = re.findall(PAT, chunk)

            # Transform into byte sequences
            byte_pre_input.extend([list(s.encode("utf-8")) for s in pre_input])

    while len(vocab) < vocab_size:
        pairs = get_pairs(byte_pre_input)
        print(pairs)
        print(vocab)
        best = max(pairs, key=lambda p: (pairs[p], vocab[p[0]], vocab[p[1]]))
        byte_pre_input = merge_pairs(best, len(vocab), byte_pre_input)
        vocab[len(vocab)] = vocab[best[0]] + vocab[best[1]]
        merges.append((vocab[best[0]], vocab[best[1]]))

    return vocab, merges


# Main program
minimal_example_path = 'bpe_example.txt'
tiny_stories_50k = 'data/TinyStoriesV2-GPT4-train_50k.txt'
tiny_stories_500k = 'data/TinyStoriesV2-GPT4-train_500k.txt'
tiny_stories = 'data/TinyStoriesV2-GPT4-train.txt'
tiny_stories_valid = 'data/TinyStoriesV2-GPT4-valid.txt'

special_tokens = ['<|endoftext|>']
vocab, merges = train_bpe(tiny_stories_500k, 258, special_tokens)
print(vocab, merges)

