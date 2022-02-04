from data.processor import KGProcessor
import os
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer 
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

class MultiprocessingEncoder(object):
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

    def initializer(self):
        global bpe
        if "pretrained" in self.model_name_or_path:
            bpe = BertTokenizer.from_pretrained(self.model_name_or_path, add_prefix_space=True)
            bpe.bos_token = bpe.cls_token
            bpe.eos_token = bpe.sep_token
        else:
            bpe = AutoTokenizer.from_pretrained(self.model_name_or_path, add_prefix_space=True)
        # self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path, add_prefix_space=True)
        reverse_list = ["(reverse)"]
        num_added_tokens = bpe.add_special_tokens({'additional_special_tokens': reverse_list})
        # num_added_tokens = bpe.add_special_tokens({'additional_special_tokens': reverse_list})

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

# 将三元组转化为文本输入，每一个三元组一行文本
# 之后使用tokenizer tokenize

def main():
    processor = KGProcessor()
    data_dir = "./dataset/AliOpenKG500"
    for mode in ["train"]:
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                end = temp[1]#.find(',')
                if "wiki" in data_dir:
                    assert "Q" in temp[0]
                ent2text[temp[0]] = temp[1] #[:end]
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                end = temp[1]#.find(',')
                if "wiki" in data_dir:
                    assert "P" in temp[0]
                rel2text[temp[0]] = temp[1] #[:end]
        with open(os.path.join(data_dir, mode+".tsv"), "r") as f:
            triples = list(map(str.strip, f.readlines()))

        with open(os.path.join(data_dir, f"examples_{mode}.txt"), "w") as file:
            for triple in tqdm(triples):
                h,r,t = triple.split()
                if h not in ent2text or t not in ent2text or r not in rel2text: continue
                head, tail = ent2text[h], ent2text[t]
                rel = rel2text[r]
                text = f"{head} {rel} \n{tail}"
                file.write(text + "\n")
                text = f"{tail} {rel} (reverse) \n{head}"
                file.write(text + "\n")
        file_inputs = [os.path.join(data_dir, f"examples_{mode}.txt")]
        file_outputs = [os.path.join(data_dir, f"features_{mode}.txt")]

        with contextlib.ExitStack() as stack:
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-" else sys.stdin
                for input in file_inputs
            ]
            outputs = [
                stack.enter_context(open(output, "w", encoding="utf-8"))
                if output != "-" else sys.stdout
                for output in file_outputs
            ]

            if "Ali" in data_dir:
                model_name_or_path = "pretrained_model"
            else:
                model_name_or_path = "facebook/bart-base"
            encoder = MultiprocessingEncoder(model_name_or_path)
            pool = Pool(16, initializer=encoder.initializer)
            encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

            stats = Counter()
            for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                if filt == "PASS":
                    for enc_line, output_h in zip(enc_lines, outputs):
                        print(enc_line, file=output_h)
                else:
                    stats["num_filtered_" + filt] += 1
                if i % 100000 == 0:
                    print("processed {} lines".format(i), file=sys.stderr)

            for k, v in stats.most_common():
                print("[{}] filtered {} lines".format(k, v), file=sys.stderr)



split = 10
with open("./dataset/wikidata5m/features_train.txt", "r") as file:
    lines = file.readlines()
    len_chunks = len(lines)//2 // split
    for i in range(split):
        with open(f"./dataset/wikidata5m/features_train_{i}.txt", "w") as f:
            f.writelines(lines[i*len_chunks*2:(i+1)*len_chunks*2 if i != split-1 else len(lines)])


if __name__ == "__main__":
    main()
    pass
