
from os import truncate
from transformers import AutoTokenizer
from typing import Dict, List
import pickle

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


class MarisaTrie(object):
    def __init__(
        self,
        sequences: List[List[int]] = [],
        cache_fist_branch=True,
        max_token_id=256001,
    ):

        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})
            assert len(self.zero_iter) == 1
            self.first_iter = list({sequence[1] for sequence in sequences})

        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )

    def get(self, prefix_sequence: List[int]):
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        elif (
            self.cache_fist_branch
            and len(prefix_sequence) == 1
            and self.zero_iter == prefix_sequence
        ):
            return self.first_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list(
                {
                    self.char2int[e[len(key)]]
                    for e in self.trie.keys(key)
                    if len(e) > len(key)
                }
            )

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)


class DummyTrieMention(object):
    def __init__(self, return_values):
        self._return_values = return_values

    def get(self, indices=None):
        return self._return_values


class DummyTrieEntity(object):
    def __init__(self, return_values, codes):
        self._return_values = list(
            set(return_values).difference(
                set(
                    codes[e]
                    for e in (
                        "start_mention_token",
                        "end_mention_token",
                        "start_entity_token",
                    )
                )
            )
        )
        self._codes = codes

    def get(self, indices, depth=0):
        if len(indices) == 0 and depth == 0:
            return self._codes["end_mention_token"]
        elif len(indices) == 0 and depth == 1:
            return self._codes["start_entity_token"]
        elif len(indices) == 0:
            return self._return_values
        elif len(indices) == 1 and indices[0] == self._codes["end_entity_token"]:
            return self._codes["EOS"]
        else:
            return self.get(indices[1:], depth=depth + 1)


import os
from tqdm import tqdm
def get_trie(args, tokenizer):
    path = os.path.join(args.data_dir, "cached_trie.pkl")
    if not args.overwrite_cache and os.path.exists(path):
        print("loading trie")
        with open(path, "rb") as file:
            trie = pickle.load(file)
            return trie
    
    if args.use_label_type:
        d = "entity2text_withtype.txt"
    else:
        d = "entity2text.txt"
    with open(f"{args.data_dir}/{d}", "r") as file:
        idx = 0
        total_entity_ids = [] 
        num_error_ids = 0
        lines = file.readlines()
        for line in tqdm(lines, desc="tokenize the prefix tree"):
            entity_name = line.split("\t")[1].strip()
            try:
                if args.output_full_sentence:
                    entity_ids = tokenizer(
                        " } [ " + entity_name + " ] ",
                        max_length=args.max_seq_length-2,
                        truncation=True
                    ).input_ids
                else:
                    entity_ids = tokenizer(" "+entity_name, add_special_tokens=True, max_length=args.max_seq_length,truncation=True).input_ids
            except:
                num_error_ids += 1
            import IPython
            # assert entity_ids not in total_entity_ids, print(entity_name)
            total_entity_ids.append(entity_ids)
            # show the entities
            if idx < 5:  print(entity_name, total_entity_ids[-1])
            idx += 1
        max_len = max(len(_) for _ in total_entity_ids)
        print("*"*10 + f"max output length : {max_len}" + "*"*10)
        print(f"total {num_error_ids} cannot be tokenized")
        
        # add </s> and <
        eos_id = tokenizer.eos_token_id
        if hasattr(args, "tgt_lang"):
            eos_id = args.bos_token_id
        if "bart" in args.model_name_or_path or "pretrained_model" in args.model_name_or_path:
            if args.output_full_sentence:
                # ignore the bos 
                trie = Trie([_[1:] for _ in total_entity_ids])
            else:
                trie = Trie([[eos_id] + _ for _ in total_entity_ids])
        else:
            # t5 unbelieveable
            trie = Trie([[0,32099] + _ for _ in total_entity_ids])
    
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


def get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )

# 太秀了 直接 {answer} [实体]
def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):

    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    codes = {
        n: encode_fn(" {}".format(c))[1]
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id

    if mention_trie is None:
        mention_trie = DummyTrieMention(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )

    if candidates_trie is None and mention_to_candidates_dict is None:
        candidates_trie = DummyTrieEntity(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ],
            codes,
        )

    sent_origs = [[codes["EOS"]] + encode_fn(sent)[1:] for sent in sentences]

    def prefix_allowed_tokens_fn(batch_id, sent):

        # 将tensor 转化为list
        sent = sent.tolist()
        #  得到status的结果， o 表示 (之前， m 表示 要生成mention 之后表示要生成entity
        status = get_status(sent)
        # sent_orig 是 原始的输入
        sent_orig = sent_origs[batch_id]
        
        # copy 机制，一对一输出

        if status == "o":
            trie_out = get_trie_outside(sent, sent_orig)
        elif status == "m":
            trie_out = get_trie_mention(sent, sent_orig)
        elif status == "e":
            trie_out = get_trie_entity(sent, sent_orig)
            if trie_out == codes["EOS"]:
                trie_out = get_trie_outside(sent, sent_orig)
        else:
            raise RuntimeError

        return trie_out

    def get_status(sent):
        c = [
            codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            )
        ]
        # e = sentence,  token 是 st ed的和，
        status = sum(e in c for e in sent) % 4

        # 如果没有 就是copy 模式，如果1 就是在里面mention 如果是2,3就是 mention 模式
        if status == 0:
            return "o"
        elif status == 1:
            return "m"
        else:
            return "e"

    def get_trie_outside(sent, sent_orig):
        # sent 当前输出  sent_orig 原始输出
        pointer_end = get_pointer_end(sent, sent_orig)

        # 如果候选项在mention trie中的话 就走 {
        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"] and sent_orig[
                pointer_end
            ] in mention_trie.get([]):
                return [sent_orig[pointer_end], codes["start_mention_token"]]
            else:
                return [sent_orig[pointer_end]]
        else:
            return []

    def get_pointer_end(sent, sent_orig):
        # 返回j在原始输入的end
        i = 0
        j = 0
        while i < len(sent):
            if sent[i] == sent_orig[j]:
                i += 1
                j += 1
            elif (
                sent[i] == codes["start_mention_token"]
                or sent[i] == codes["end_mention_token"]
            ):
                i += 1
            elif sent[i] == codes["start_entity_token"]:
                i += 1
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None

    def get_trie_mention(sent, sent_orig):

        pointer_start, _ = get_pointer_mention(sent)
        if pointer_start + 1 < len(sent):
            ment_next = mention_trie.get(sent[pointer_start + 1 :])
        else:
            ment_next = mention_trie.get([])

        pointer_end = get_pointer_end(sent, sent_orig)

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"]:
                if sent_orig[pointer_end] in ment_next:
                    if codes["EOS"] in ment_next:
                        return [sent_orig[pointer_end], codes["end_mention_token"]]
                    else:
                        return [sent_orig[pointer_end]]
                elif codes["EOS"] in ment_next:
                    return [codes["end_mention_token"]]
                else:
                    return []
            else:
                return [codes["end_mention_token"]]
        else:
            return []

    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    def get_trie_entity(sent, sent_orig):
        pointer_start, pointer_end = get_pointer_mention(sent)

        if pointer_start + 1 != pointer_end:
            mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()

            if candidates_trie is not None:
                candidates_trie_tmp = candidates_trie
            elif mention_to_candidates_dict is not None:
                candidates_trie_tmp = Trie(
                    [
                        encode_fn(
                            " {} {} {} {}".format(
                                end_mention_token,
                                start_entity_token,
                                e,
                                end_entity_token,
                            )
                        )[1:]
                        for e in mention_to_candidates_dict.get(mention, ["NIL"])
                    ]
                )
            else:
                raise RuntimeError()

            return candidates_trie_tmp.get(sent[pointer_end:])

        return []

    return prefix_allowed_tokens_fn

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
    model_real_name = model_name.split("/")[-1]
    with open(f"{model_real_name}_{dataset}.pkl", "wb") as file:
        pickle.dump(trie, file)
    


