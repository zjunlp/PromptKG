
from transformers import BartForConditionalGeneration, AutoTokenizer, GPT2LMHeadModel, BartModel, T5ForConditionalGeneration,AutoModelForSeq2SeqLM, AutoConfig
import torch
import csv
from collections import defaultdict, Counter
relation2pattern = defaultdict(list)
from tqdm import tqdm
import re

device = "cuda:1"
# model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model_name_or_path = "facebook/bart-base"
config = AutoConfig.from_pretrained(model_name_or_path)
config.force_bos_token_to_be_generated = True
config.use_prefix = False
config.preseqlen = 1
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
i = 0
model = model.to(device)
from data.processor import KGProcessor
processor = KGProcessor()
path = "dataset/FB15k-237"
examples = processor.get_train_examples(path)
def get_mask_word(text, model, tokenizer):
    origin_tokens = text.split()
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k:v.to(device) for k,v in inputs.items()}
    output = model.generate(
        **inputs, 
        num_beams=10,
        num_return_sequences=1
    )
    _txt = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # for t in origin_tokens:
    #     _txt = _txt.replace(t, "<e>")
    return _txt
for example in tqdm(examples):
    i += 1
    if i%2: continue
    head,relation,tail = example.text_a,example.text_b,example.text_c
    text = " <mask> " + head + " <mask> " + relation + " <mask> " + tail + " <mask> "
    p = get_mask_word(text, model, tokenizer)
    p = re.sub(' +', '', p)
    for t_ in [head, relation, tail]:
        p = p.replace(t_, "\t")
    p = re.sub(' +', '', p)
    relation2pattern[relation].append(p)

for k in relation2pattern.keys():
    relation2pattern[k] = Counter(relation2pattern[k])

import pickle 
import os
with open(os.path.join(path, "cached_relation_pattern.pkl"), "wb") as file:
    pickle.dump(relation2pattern, file)

pattern = {}
for k,v in relation2pattern.items():
    t = v.most_common(1)[0][0]
    t = [_.strip() for _ in t.split("\t")]
    t[-1] = " ."
    print(k)
    print(t)
    pattern[k] = t
    assert len(t) ==4

with open(os.path.join(path, "cached_relation_pattern.pkl"), "wb") as file:
    pickle.dump(pattern, file)