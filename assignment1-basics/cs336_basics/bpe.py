from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
from multiprocessing import Pool
import time
from copy import deepcopy

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)
DEBUG = False

class BPE_Tokenizer(object):
    def __init__(self, input_path, vocab_size, special_tokens: list[str], num_processes=4) -> None:
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.input_path = input_path

        self.vocab = {}
        self.freq_table = {}
        self.pairs = Counter()
        self.merges = []
        
        self.init_vocab()
        self.pretokenization(num_processes)
        self.compute_merges()

    def init_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for i, token in enumerate(self.special_tokens):
            vocab[i + 256] = token.encode("utf-8")
        self.vocab = vocab

    def pretokenization_for_one_chunk(self, start, end):
        with open(self.input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # split on special tokens
            counter = Counter()
            docs = re.split("|".join(re.escape(token) for token in self.special_tokens), chunk)
            for doc in docs:
                for match in PAT.finditer(doc):
                    counter[match.group()] += 1

        return counter

    def pretokenization(self, num_processes=4):
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # parallelize this by sending each start/end pair to a set of processes.

            # counters = []
            # for start, end in zip(boundaries[:-1], boundaries[1:]):
            #     print(start, end)
            #     counters.append(self.pretokenization_for_one_chunk(special_tokens, (start, end)))
            with Pool(processes=num_processes) as pool:
                ranges = zip(boundaries[:-1], boundaries[1:])
                counters = pool.starmap(self.pretokenization_for_one_chunk, ranges)
        
        total_counter = Counter()
        for counter in counters:
            total_counter.update(counter)

        # print("########## pre tokenization finished ##########")
        
        for token in total_counter:
            bytes_token = tuple([bytes([b]) for b in token.encode('utf-8')])
            self.freq_table[bytes_token] = total_counter[token]
            for i in range(len(bytes_token) - 1):
                self.pairs[(bytes_token[i], bytes_token[i + 1])] += total_counter[token]

        # print("########## pairs initialized ##########")

    def compute_merges_for_one_loop(self):
        max_freq = max(self.pairs.values())
        merge = max((token for token, freq in self.pairs.items() if freq == max_freq))
        
        self.pairs[merge[0], merge[1]] = 0
        new_bytes = merge[0] + merge[1]
        if DEBUG:
            print("[new merge]", new_bytes)

        self.merges.append(merge)
        self.vocab[len(self.vocab)] = new_bytes

        tokens_to_remove = []
        incremental_freq_table = {}
        for bytes_token, freq in self.freq_table.items():
            if len(bytes_token) < 2:
                continue

            if DEBUG and new_bytes == b' t':
                breakpoint()
            
            new_bytes_token = []
            i = 0
            while i < len(bytes_token) - 1:
                if bytes_token[i] == merge[0] and bytes_token[i + 1] == merge[1]:
                    new_bytes_token.append(new_bytes)
                    if i > 0:
                        self.pairs[(bytes_token[i - 1], bytes_token[i])] -= freq
                        self.pairs[(bytes_token[i - 1], new_bytes)] += freq
                    if i < len(bytes_token) - 2:
                        self.pairs[(bytes_token[i + 1], bytes_token[i + 2])] -= freq
                        self.pairs[(new_bytes, bytes_token[i + 2])] += freq
                    i += 2
                else:
                    new_bytes_token.append(bytes_token[i])
                    i += 1
                if i == len(bytes_token) - 1:
                    new_bytes_token.append(bytes_token[i])

            new_bytes_token = tuple(new_bytes_token)
            if new_bytes_token != bytes_token:
                tokens_to_remove.append(bytes_token)
                if DEBUG:
                    print(new_bytes_token, bytes_token)
                incremental_freq_table[new_bytes_token] = freq

        for token in tokens_to_remove:
            self.freq_table.pop(token)
        self.freq_table.update(incremental_freq_table)

    def compute_merges(self):
        i = 0
        while len(self.vocab) < self.vocab_size:
            self.compute_merges_for_one_loop()
            if DEBUG:
                i += 1
                breakpoint()
                print(f"########## compute the {i}th merge for one loop ##########")

if __name__ == '__main__':
    input_path = "/root/autodl-tmp/a1-data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 1000
    t = time.time()
    tokenizer = BPE_Tokenizer(input_path, vocab_size, special_tokens)
    # print(type(list(tokenizer.pretokens.keys())[0]))
    print(time.time() - t)