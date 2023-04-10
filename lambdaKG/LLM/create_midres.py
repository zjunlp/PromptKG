#if you need to generate the MidRes.json by yourself, the following code may help you instruct smoothly.
import json
entity2textlong ={}
corpus=[]
middle_reason={}
entity2text={}
relation={}


fr1=open("dataset/FB15k-237/entity2text.txt")
for line in fr1.readlines():
    lines=line.strip('\n').split('\t')
    entity2text[lines[0]]=lines[1][:lines[1].find(',')]
    entity2textlong[lines[0]]=lines[1][lines[1].find(',')+2:]
    
fr2=open("dataset/FB15k-237/relations.txt")
c=0
for line in fr2.readlines():
    relation[str(c)]= line.strip('\n')
    c+=1

f=open("dataset/FB15k-237/train.tsv")

def build_midres():
    for line in f.readlines():
        lines= line.strip('\n').split('\t')
        rel = relation[lines[1]].split('/')[-1].replace('_',' ')
        corpus.append("what is the {} of {}? The answer is {}.\n".format(rel,entity2text[lines[0]],entity2text[lines[2]]))
        find_relv=False
        relv_text=''
        for i in entity2textlong[lines[0]].replace('\\n',' ').split('. '):
            if relv_text=='' and entity2text[lines[0]].lower() in i.lower():
                relv_text+=i+'. '
            if (entity2text[lines[2]]).lower() in i.lower() :
                find_relv=True
                relv_text+=' '+i+'.'
        if find_relv is False:
            relv_text=' '+entity2textlong[lines[0]].split('. ')[0]+'.'
        middle_reason["what is the {} of {}? The answer is {}.\n".format(rel,entity2text[lines[0]],entity2text[lines[2]])]=relv_text

wt=open("dataset/FB15k-237/MidRes.json",'w')
build_midres()
jstr=json.dumps(middle_reason)
json.dump(jstr,wt)
