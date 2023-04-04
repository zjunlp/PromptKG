import json
import jsonlines
from collections import defaultdict

data_grouped_by_query = defaultdict(set)

with open('dataset/atomic_2020_data/test.tsv') as fp:
    test_data = fp.readlines()
    for line in test_data:
        line_data = line.strip().split('\t')
        query = f'{line_data[0]} @@ {line_data[1]}'
        data_grouped_by_query[query].add(line_data[2])

all_data = []
with open('dataset/atomic_2020_data/test_process.tsv', 'w') as fp:
    for k, v in data_grouped_by_query:
        all_data.append(f"{k}\t{'| '.join(list(v))}\n")
    fp.writelines(all_data)
    
with open("LLM/sample.json") as fp:
    sample = json.load(fp)
    
with jsonlines.open('dataset/atomic_2020_data/test.jsonl', 'w') as fp:
    for line in all_data:
        rel = line.strip().split('\t')[0].split("@@")[1].strip()
        examples = sample[rel]
        one_test = dict(query=line.split('\t')[0], examples=examples, answer=line.strip().split('\t')[1].split('| '))
        fp.write(one_test)
        
