from typing import List
import inspect
import unicodedata
import urllib
import torch
import numpy as np
import json
import pickle
from collections import deque



class LinkGraph:

    def __init__(self, examples):
        self.graph = {}
        for ex in examples:
            head_id, tail_id = ex.hr
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[int]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_n_hop_entity_indices(self, entity_id: int,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return list(seen_eids)


# def cache_results(_cache_fp, _refresh=False, _verbose=1):
#     r"""
#     cache_results是fastNLP中用于cache数据的装饰器。通过下面的例子看一下如何使用::
#         import time
#         import numpy as np
#         from fastNLP import cache_results
        
#         @cache_results('cache.pkl')
#         def process_data():
#             # 一些比较耗时的工作，比如读取数据，预处理数据等，这里用time.sleep()代替耗时
#             time.sleep(1)
#             return np.random.randint(10, size=(5,))
        
#         start_time = time.time()
#         print("res =",process_data())
#         print(time.time() - start_time)
        
#         start_time = time.time()
#         print("res =",process_data())
#         print(time.time() - start_time)
        
#         # 输出内容如下，可以看到两次结果相同，且第二次几乎没有花费时间
#         # Save cache to cache.pkl.
#         # res = [5 4 9 1 8]
#         # 1.0042750835418701
#         # Read cache from cache.pkl.
#         # res = [5 4 9 1 8]
#         # 0.0040721893310546875
#     可以看到第二次运行的时候，只用了0.0001s左右，是由于第二次运行将直接从cache.pkl这个文件读取数据，而不会经过再次预处理::
#         # 还是以上面的例子为例，如果需要重新生成另一个cache，比如另一个数据集的内容，通过如下的方式调用即可
#         process_data(_cache_fp='cache2.pkl')  # 完全不影响之前的‘cache.pkl'
#     上面的_cache_fp是cache_results会识别的参数，它将从'cache2.pkl'这里缓存/读取数据，即这里的'cache2.pkl'覆盖默认的
#     'cache.pkl'。如果在你的函数前面加上了@cache_results()则你的函数会增加三个参数[_cache_fp, _refresh, _verbose]。
#     上面的例子即为使用_cache_fp的情况，这三个参数不会传入到你的函数中，当然你写的函数参数名也不可能包含这三个名称::
#         process_data(_cache_fp='cache2.pkl', _refresh=True)  # 这里强制重新生成一份对预处理的cache。
#         #  _verbose是用于控制输出信息的，如果为0,则不输出任何内容;如果为1,则会提醒当前步骤是读取的cache还是生成了新的cache
#     :param str _cache_fp: 将返回结果缓存到什么位置;或从什么位置读取缓存。如果为None，cache_results没有任何效用，除非在
#         函数调用的时候传入_cache_fp这个参数。
#     :param bool _refresh: 是否重新生成cache。
#     :param int _verbose: 是否打印cache的信息。
#     :return:
#     """

#     def wrapper_(func):
#         signature = inspect.signature(func)
#         for key, _ in signature.parameters.items():
#             if key in ('_cache_fp', '_refresh', '_verbose'):
#                 raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

#         def wrapper(*args, **kwargs):
#             my_args = args[0]
#             mode = args[-1]
#             if '_cache_fp' in kwargs:
#                 cache_filepath = kwargs.pop('_cache_fp')
#                 assert isinstance(cache_filepath, str), "_cache_fp can only be str."
#             else:
#                 cache_filepath = _cache_fp
#             if '_refresh' in kwargs:
#                 refresh = kwargs.pop('_refresh')
#                 assert isinstance(refresh, bool), "_refresh can only be bool."
#             else:
#                 refresh = _refresh
#             if '_verbose' in kwargs:
#                 verbose = kwargs.pop('_verbose')
#                 assert isinstance(verbose, int), "_verbose can only be integer."
#             else:
#                 verbose = _verbose
#             refresh_flag = True
            
#             model_name = my_args.model_name_or_path.split("/")[-1]
#             is_pretrain = my_args.pretrain
#             cache_filepath = os.path.join(my_args.data_dir, f"cached_{mode}_features{model_name}_pretrain{is_pretrain}_faiss{my_args.faiss_init}_seqlength{my_args.max_seq_length}_{my_args.litmodel_class}.pkl")
#             refresh = my_args.overwrite_cache

#             if cache_filepath is not None and refresh is False:
#                 # load data
#                 if os.path.exists(cache_filepath):
#                     with open(cache_filepath, 'rb') as f:
#                         results = pickle.load(f)
#                     if verbose == 1:
#                         logger.info("Read cache from {}.".format(cache_filepath))
#                     refresh_flag = False

#             if refresh_flag:
#                 results = func(*args, **kwargs)
#                 if cache_filepath is not None:
#                     if results is None:
#                         raise RuntimeError("The return value is None. Delete the decorator.")
#                     with open(cache_filepath, 'wb') as f:
#                         pickle.dump(results, f)
#                     logger.info("Save cache to {}.".format(cache_filepath))

#             return results

#         return wrapper

#     return wrapper_

class Roberta_utils(object):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.modL = 7
        self.max_sentence_length = 200
        embed_path = './dataset/wikiembed_roberta/'
        with open(embed_path+'name2id.json') as fp:
            self.name2pageid =  json.load(fp)
            print("Successfully load name2id.json.")

        with open(embed_path +'qid2pageid.json') as fp:
            self.qid2pageid =  json.load(fp)
            print("Successfully load qid2pageid.json.")

        with open(embed_path+'wiki_pageid2embedid.pkl', 'rb') as fp:
            self.pageid2id = pickle.load(fp)
            print("Successfully load wiki_pageid2embedid.pkl.")

        with open(embed_path+'qid2embedid.pkl', 'rb') as fp:
            self.qid2embedid = pickle.load(fp)
            print("Successfully load qid2embedid.pkl.")
        
        self.tot_entity_embed = np.load(embed_path + 'wiki_entity_embed_256.npy')
        print("Successfully load wiki_entity_embed_256.npy.")
        L = np.linalg.norm(self.tot_entity_embed, axis=1)
        self.tot_entity_embed = self.tot_entity_embed / np.expand_dims(L, axis=-1) * self.modL
        
        self.dim = self.tot_entity_embed.shape[1]

    def _is_subword(self, token):
        token = self.tokenizer.convert_tokens_to_string(token)
        if not token.startswith(" ") and not self._is_punctuation(token[0]):
            return True
        return False

    @staticmethod
    def _is_punctuation(char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
    
    @staticmethod
    def _normalize_mention(text):
        return " ".join(text.split(" ")).strip()


    def _detect_mentions(self, tokens, name):
        name = self._normalize_mention(name)

        for start, token  in enumerate(tokens):

            if self._is_subword(token) and (start>1):
                continue
        
            for end in range(start, len(tokens)):
                if end < len(tokens) and self._is_subword(tokens[end]) and (end>1):
                    continue
                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = self._normalize_mention(mention_text)
                if len(mention_text) > len(name):
                    break
                if mention_text.lower()==name.lower():
                    return start, end


        return -1, -1

    def _detech_mentions_squad(self, tokens, ent2id):
        mentions = []
        cur = 0
        for start, token  in enumerate(tokens):
            if start < cur:
                continue

            if self._is_subword(token) and (start>1):
                continue

            for end in range(min(start + 30, len(tokens)), start, -1):
                if end < len(tokens) and self._is_subword(tokens[end]) and (end>1):
                    continue

                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = self._normalize_mention(mention_text)
                if mention_text in ent2id:
                    cur = end
                    pageid = ent2id[mention_text]
                    mentions.append((pageid, start, end))
                    break
        return mentions

    def get_batch(self, sentences_list, sub_labels=None, sub_ids=None):
        # try_cuda = False   # for debug
        if not sentences_list:
            return None

        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        input_embeds_list = []
        attention_mask_list = []
        position_ids_list = []
        input_ids_list = []

        entity_embeddings_list = []
        entity_attention_mask_list = []
        entity_position_ids_list = []

        entity_K = 1
        if sub_ids is None:
            sub_ids = [-1]* len(sentences_list)
        for masked_inputs_list, sub_label, sub_id in zip(sentences_list, sub_labels, sub_ids):
            #assert len(masked_inputs_list)!=1
            for idx, masked_input in enumerate(masked_inputs_list):
                if sub_id in self.qid2pageid.keys():
                    sub_pageid = self.qid2pageid[sub_id]
                else:
                    sub_label_align = urllib.parse.unquote(sub_label) # Added
                    if sub_label_align in self.name2pageid.keys():
                        sub_pageid =  self.name2pageid[sub_label_align]
                    elif sub_label_align.lower() in self.name2pageid.keys():
                        sub_pageid =  self.name2pageid[sub_label_align.lower()]
                    else:
                        sub_pageid = -1
                # print('masked_input', masked_input)
                tokens = self.tokenizer.tokenize(masked_input)
                tokens = [self.tokenizer.cls_token] + tokens #+ [self.tokenizer.sep_token]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask_s = -1
                for k in range(len(tokens)):
                    if tokens[k]=='<mask>':
                        mask_s = k
                        break
                assert(mask_s!=-1)
                if input_ids[mask_s-1]==1437:
                    input_ids = input_ids[:mask_s-1] + input_ids[mask_s:]
                    tokens = tokens[:mask_s-1] + tokens[mask_s:]
                    mask_s -= 1

                l_id = self.tokenizer.encode(' (', add_special_tokens=False)
                assert(len(l_id)==1)
                r_id = self.tokenizer.encode(' )', add_special_tokens=False)
                assert(len(r_id)==1)


                output_tokens = []
                mentions = []

                if sub_label=='Squad':
                    assert (False)
                    ents = self.qid2ents[sub_id]
                    ent2id = {}
                    for x in ents:
                        if x[1]!='MASK':
                            ent2id[self._normalize_mention(x[1])] = x[0]
                    mentions = self._detech_mentions_squad(tokens, ent2id)

                elif  sub_pageid>=0 and self.modL>0:
                    sub_s, sub_e = self._detect_mentions(tokens, sub_label)
                    if( sub_s>=0):

                        mentions = [(sub_pageid, sub_s, sub_e, sub_e+1)]
                        input_ids = input_ids[:sub_e] + l_id + [self.tokenizer.mask_token_id] + r_id + input_ids[sub_e:]
                        mask_s += 3

                    else:
                        print (tokens, sub_label)
                        mentions = []

                input_ids = input_ids[:self.max_sentence_length-1] + [self.tokenizer.sep_token_id]
                entity_embeddings = []
                entity_position_ids = []
                # entity_attention_mask = []
                np.random.shuffle(mentions)

                for page_id, sub_s, sub_e, pos_ent in mentions:
                    if page_id in self.pageid2id:
                        embed_id = self.pageid2id[page_id]
                        try:
                            entity_embedding = np.array(self.tot_entity_embed[embed_id], dtype=np.float32)

                            entity_embeddings.append(entity_embedding)
                            entity_position_ids.append(pos_ent + 2)
                        except:
                            pass
                    if len(entity_embeddings)>=entity_K:
                        break

                L = len(input_ids)
                max_len = max(max_len, L)

                padding_length = self.max_sentence_length - L
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * L + [0] * padding_length
                attention_mask += [1] * len(entity_position_ids) + [0] * (entity_K-len(entity_position_ids))
                for page_id, sub_s, sub_e, pos_ent in mentions:
                    attention_mask[pos_ent] = 0



                while len(entity_embeddings) < entity_K:
                    entity_embeddings.append(np.zeros((self.dim, ), dtype=np.float32))
                    entity_position_ids.append(0)

                output_tokens_list.append(np.array(input_ids, dtype=np.int64))

                input_ids_list.append(input_ids)

                attention_mask_list.append(attention_mask)
                masked_indices_list.append(mask_s)
                entity_embeddings_list.append(torch.tensor(np.array(entity_embeddings, dtype=np.float32), dtype=torch.float32))
                entity_position_ids_list.append(entity_position_ids)


        input_ids_list = torch.tensor(np.array(input_ids_list, dtype=np.int64), dtype=torch.int64)
        attention_mask_list = torch.tensor(np.array(attention_mask_list, dtype=np.int64), dtype=torch.int64)
        masked_indices_list = torch.tensor(np.array(masked_indices_list, dtype=np.int64), dtype=torch.int64)
        entity_embeddings_list = torch.stack(entity_embeddings_list,  dim=0)
        entity_position_ids_list = torch.tensor(np.array(entity_position_ids_list, dtype=np.int64),  dtype=torch.int64)

        return input_ids_list, attention_mask_list, entity_embeddings_list, entity_position_ids_list, masked_indices_list