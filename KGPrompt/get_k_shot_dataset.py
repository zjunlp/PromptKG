import os 
import csv
import sys
import random
import time
import os



dataset = "FB15k-237"
k = 8
seed = 666

train_path = f"./dataset/{dataset}/train.tsv"
# train_path = "data/WN18RR-8shot/42/test.tsv"
random.seed(seed)

with open(train_path, "r") as f:
	reader = csv.reader(f, delimiter="\t", quotechar=None)
	lines = []
	for line in reader:
		if sys.version_info[0] == 2:
			line = list(unicode(cell, 'utf-8') for cell in line)
		lines.append(line)


# shuffle lines and pack the triples by relation
random.shuffle(lines)

triple_by_rel = {}
cnt_by_rel = {}

for line in lines:
	head, rel, tail = line
	if rel not in triple_by_rel:
		triple_by_rel[rel] = [[head, rel, tail]]
		cnt_by_rel[rel] = 1
	elif cnt_by_rel[rel] < k:
		triple_by_rel[rel].append([head, rel, tail])
		cnt_by_rel[rel] += 1

# pick k shot every rel


with open(train_path.replace("train", f"{k}shot-{seed}-train") , "w") as f:
	writer = csv.writer(f, delimiter="\t")
	for v in triple_by_rel.values():
		for tt in v:
			writer.writerow(tt)

os.mkdir(f"./dataset/{seed}")
os.system(f"cp ./dataset/{dataset}/* ./dataset/{seed} ")

os.system(f"mv ./dataset/{seed} ./dataset/{dataset}")
os.system(f"mv ./dataset/{dataset}/{k}shot-{seed}-train.tsv train.tsv ./dataset/{dataset}/{seed}")
os.system(f"mv ./dataset/{dataset}/{seed}/{k}shot-{seed}-train.tsv train.tsv")
