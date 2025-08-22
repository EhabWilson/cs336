from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
from multiprocessing import Pool
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer(object):
    def __init__(self, input_path, special_tokens: list[str]) -> None:
        self.special_tokens = special_tokens
        self.input_path = input_path

        self.init_vocab()
        self.pretokenization()

    def init_vocab(self):
        vocab = {i: chr(i).encode("utf-8") for i in range(256)}
        for i, token in enumerate(self.special_tokens):
            vocab[i + 256] = token.encode("utf-8")
        self.vocab = vocab

    def pretokenization_for_one_chunk(self, special_tokens: list[str], chunk_range):
        with open(self.input_path, "rb") as f:
            start, end = chunk_range
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # split on special tokens
            counter = Counter()
            docs = re.split("|".join(special_tokens), chunk)
            for doc in docs:
                for match in re.finditer(PAT, doc):
                    counter[match.group()] += 1

        return counter

    def pretokenization(self, num_processes=4):
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # parallelize this by sending each start/end pair to a set of processes.
            special_tokens = [re.escape(token) for token in self.special_tokens]

            # counters = []
            # for start, end in zip(boundaries[:-1], boundaries[1:]):
            #     print(start, end)
            #     counters.append(self.pretokenization_for_one_chunk(special_tokens, (start, end)))
            with Pool(processes=num_processes) as pool:
                ranges = list(zip(boundaries[:-1], boundaries[1:]))
                counters = pool.starmap(self.pretokenization_for_one_chunk, zip([special_tokens]*num_processes, ranges))
        
        total_counter = Counter()
        for counter in counters:
            total_counter.update(counter)
        
        self.pretokens = total_counter

def compute_merges_for_one_loop():
    pass

if __name__ == '__main__':
    input_path = "/root/autodl-tmp/a1-data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    t = time.time()
    tokenizer = BPE_Tokenizer(input_path, special_tokens)
    # result = pretokenization(input_path, special_tokens)
    print(time.time() - t)