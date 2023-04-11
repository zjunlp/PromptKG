from rank_bm25 import BM25Okapi
import random
import openai
import json
import time
import argparse
import os
from easyinstruct import BasePrompt
from easyinstruct.utils import set_openai_key, set_proxy

instruct_prompt=f"given head entity and relation, predict the tail entity from the candidates:"
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

set_openai_key(api_key)
# set_proxy("http://127.0.0.1:7890")

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate_num", type=int, default=100)
    parser.add_argument("--demonstrations_num", type=int, default=5)
    parser.add_argument("--entity_path",default="dataset/FB15k-237/entity2text.txt")
    parser.add_argument("--relation_path",default="dataset/FB15k-237/relations.txt")
    parser.add_argument("--corpus_path",default="dataset/FB15k-237/train.tsv")
    parser.add_argument("--rationale_path",default="dataset/FB15k-237/MidRes.json")
    parser.add_argument("--path",default="dataset/FB15k-237/test.tsv")
    parser.add_argument("--help", "-h", action="help") 

    return parser


def load_entity(fpath):
    #entity -> text entity2text{ID:txt}
    fr2=open(fpath,'r',encoding='utf-8')
    entity2text={}
    for line in fr2.readlines():
        lines=line.strip('\n').split('\t')
        entity2text[lines[0]]=lines[1].split(',')[0]

    return entity2text

def load_relation(fpath):
    fr2=open(fpath,encoding='utf-8')
    relation={}
    c=0
    for line in fr2.readlines():
        relation[str(c)]= line.strip('\n')
        c+=1
    return relation

def load_corpus(fpath,relation,entity2text):
    # use the train set as the corpus
    corpus=[]
    f=open(fpath,'r')
    for line in f.readlines():
        lines= line.strip('\n').split('\t')
        rel = relation[lines[1]].split('/')[-1].replace('_',' ')
        corpus.append("what is the {} of {}? The answer is {}.\n".format(rel,entity2text[lines[0]],entity2text[lines[2]]))
    return corpus

def load_rationales(fpath):
    # MidRes.json includes the important rationales from the train set.
    middle_reason={}
    f=open(fpath,'r')
    jstr=json.load(f)
    middle_reason=json.loads(jstr)
    return middle_reason

def find_bm25(corpus):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def run(args):
    candidate_num=args.candidate_num
    demonstrations_num=args.demonstrations_num
    entity2text=load_entity(args.entity_path)
    relations = load_relation(args.relation_path)
    corpus=load_corpus(args.corpus_path,relations,entity2text)
    bm25=find_bm25(corpus)
    middle_reason=load_rationales(args.rationale_path)

    f=open(args.path,'r')

    count=0
    total=0
    line = f.readline()
    
    while line:
        total+=1
        items=line.strip('\n').split('\t')
        rel = relations[items[1]].split('/')[-1].replace('_',' ')

        query = "what is the {} of {}?".format(rel,entity2text[items[0]])

        tokenized_query = query.split(" ")
        select_cropus=bm25.get_top_n(tokenized_query, corpus, n=candidate_num+50)

        # select demonstrations for input data
        demos= select_cropus[:demonstrations_num]
        quest_demos=[i[:i.find('?')+1] for i in demos]
        answer_demos=[i[i.find("?")+1:] for i in demos]
        middle_reasons=[middle_reason[i] for i in demos]
        final_demos=[quest_demos[i]+middle_reasons[i]+answer_demos[i] for i in range(demonstrations_num)]

        # select candidates for input data
        candidates=[i[i.find('The answer is ')+14:i.find('.\n')]for i in select_cropus]
        candidates=list(set(candidates))
        if len(candidates)<candidate_num:
            candidates+=random.sample(list(entity2text.values()),candidate_num-len(candidates))
        elif len(candidates)>candidate_num:
            candidates=candidates[:candidate_num]
        random.shuffle(candidates)

        # concate the instruct_prompt, candidates, demonstrations and the input question as prompt
        prompt=instruct_prompt+", ".join(candidates)+'.\n'+"".join(final_demos)+"what is the {} of {}?".format(rel,entity2text[items[0]])

        # show the prompt and answer for input data
        # print(f"prompt: {prompt}")
        # print(f"answer: {entity2text[items[2]]}")

        count=test_on_gpt3(prompt,entity2text[items[2]],count)
        if total%1000==0: 
            print(f"processing the {total} item, current hit@1 is {count/total}")

        line = f.readline()
    
    print(f"hit@1 on {args.path} is {count/total}")
        

def test_on_gpt3(prompt,answer,count):
    try:
        # use easyinstruct from https://github.com/zjunlp/EasyInstruct
        promts = BasePrompt()
        promts.build_prompt(prompt)
        response = promts.get_openai_result(engine = "text-davinci-003", temperature = 0.5, max_tokens = 200)
        # response = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=prompt,
        #     temperature=0.5,
        #     max_tokens=200,
        #     top_p=1,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1
        #     )
        time.sleep(1.5)
    except:
        raise Exception("An error occurred while calling the OpenAI API for text-davinci-003! ")

    if answer in response['choices'][0]['text']:
        count+=1
        
    return count


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    run(args)
