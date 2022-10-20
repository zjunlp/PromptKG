from os import truncate
from transformers import AutoTokenizer
from typing import Dict, List
import pickle
import json

try:
    import marisa_trie
except ModuleNotFoundError:
    pass


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)
    
    def del_entity(self, sequence):
        root = self.trie_dict
        t_root = root
        pop_ids = sequence[0]
        s_idx = 0
        while s_idx < len(sequence)-1:
            if len(root.keys()) > 1:
                t_root = root
                pop_ids = sequence[s_idx]
            root = root[sequence[s_idx]]
            s_idx += 1

        try:
            t_root.pop(pop_ids)
            self.len -= 1
        except:
            print("not in the trie")

import os
from tqdm import tqdm
def get_trie(args, tokenizer):
    path = os.path.join(f"dataset/{args.dataset}", "cached_trie.pkl")
    # if not args.overwrite_cache and os.path.exists(path):
    #     print("loading trie")
    #     with open(path, "rb") as file:
    #         trie = pickle.load(file)
    #         print("first 5 sequences are ---")
    #         for i, x in enumerate(trie):
    #             print(x)
    #             if i == 5:
    #                 break

    #         return trie
    
    d = "entity2text.txt"
    with open(f"dataset/{args.dataset}/{d}", "r") as file:
        idx = 0
        total_entity_ids = [] 
        num_error_ids = 0
        lines = file.readlines()


        for line in tqdm(lines, desc="tokenize the prefix tree"):
            line = line.strip()
            if "kg" in d:
                h,r,t = line.split("\t")
                entity_name = tokenizer.sep_token.join([h,r,t])
            else:
                _, text = line.split("\t")
                text = text.split(",")[0]
                entity_name = text
            try:
                entity_ids = tokenizer(entity_name, add_special_tokens=True, max_length=args.max_seq_length,truncation=True).input_ids
            except:
                num_error_ids += 1
            # assert entity_ids not in total_entity_ids, print(entity_name)
            total_entity_ids.append(entity_ids)
            # show the entities
        max_len = max(len(_) for _ in total_entity_ids)
        print("*"*10 + f"max output length : {max_len}" + "*"*10)
        print(f"total {num_error_ids} cannot be tokenized")
        
        # add </s> and <
        eos_id = tokenizer.eos_token_id
        if "bart" in args.model_name_or_path or "pretrained_model" in args.model_name_or_path:
            trie = Trie([[eos_id] + _ for _ in total_entity_ids])
        else:
            # t5 unbelieveable
            trie = Trie([[tokenizer.pad_token_id] + _ for _ in total_entity_ids])
        for i, x in enumerate(trie):
            print(x)
            if i == 5: break

    
    with open(path, "wb") as file:
        pickle.dump(trie, file)
    return trie


from typing import Dict, List

import torch

# from genre.trie import DummyTrieEntity, DummyTrieMention, Trie



def get_end_to_end_prefix_allowed_tokens_fn_hf(
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    tokenizer=None
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: tokenizer.encode(x),
        lambda x: tokenizer.decode(torch.tensor(x)),
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        len(tokenizer) - 1,
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )

if __name__ == "__main__":
    total_entity_ids = []
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = "FB15k-237"
    with open(f"../dataset/{dataset}/entity2text.txt", "r") as file:
        idx = 0
        for line in file.readlines():
            entity_name = line.split("\t")[-1].strip()
            total_entity_ids.append(tokenizer(entity_name, add_special_tokens=True).input_ids)
            # show the entities
            if idx < 5:  print(entity_name, total_entity_ids[-1])
            idx += 1
    
    # add </s> and <
    trie = Trie([_[1:] for _ in total_entity_ids])
    s = []
    for i, t in enumerate([_[1:] for _ in total_entity_ids]):
        if t[:2] == [620, 8110, 6, 2705]:
            s.append(i)
    print(s)
    # import IPython; IPython.embed(); exit(1)
    # a = [[1,2,3,4,0], [1,2,3,0], [2,3,4,0],[1,2,3,4,5,0]]
    # trie = Trie(a)
    for t in [_[1:] for _ in total_entity_ids]:
        try:
            trie.del_entity(t)
        except:
            print("e")
        # model_real_name = model_name.split("/")[-1]
        # with open(f"{model_real_name}_{dataset}.pkl", "wb") as file:
        #     pickle.dump(trie, file)
        