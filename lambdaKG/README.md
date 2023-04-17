<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/lambda_logo.png" width="550px">
    <p> <b>
        A Library for Pre-trained Language Model-Based Knowledge Graph Embeddings.</b>
    </p>
    
------

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#run-your-first-experiment">How To Use</a> •
  <a href="https://arxiv.org/abs/2210.00305">Paper</a> •
  <a href="https://medium.com/@jack16900/lambdakg-a-library-for-pre-trained-language-model-based-knowledge-graph-embeddings-728d15c6f994">Medium</a> •
  <a href="#citation">Citation</a> •
  <a href="#other-kg-representation-open-source-projects">Others</a> 
</p>
</div>

## Overview
Knowledge Graphs (KGs) often have two characteristics: heterogeneous graph structure and text-rich entity/relation information. Text-based KG embeddings can represent entities by encoding descriptions with pre-trained language models, but no open-sourced library is specifically designed for KGs with PLMs at present. 

We present **LambdaKG**, a library for KGE that equips with many pre-trained language models (e.g., BERT, BART, T5, GPT-3), and supports various tasks (e.g., knowledge graph completion, question answering, recommendation, and knowledge probing). 

**LambdaKG** is now publicly open-sourced, with [a demo video](https://www.bilibili.com/video/BV1ZY4y1C7bt/) and long-term maintenance.
<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/framework-1.png" width="750px">
</div>

## Quick Start

### Installation

**Step1** Download the basic code

```bash
git clone --depth 1 https://github.com/zjunlp/PromptKG.git
```

**Step2** Create a virtual environment using `Anaconda` and enter it.<br>


```bash
conda create -n lambdakg python=3.8

conda activate lambdakg
```
   
**Step3** Enter the task directory and install library

```bash
cd PrompKG/lambdaKG

pip install -r requirements.txt
```

#### Step4

Install our preprocessed datasets and put them into the `dataset` folder.

| Dataset (KGC) | Google Drive                                                 | Baidu Cloud |
| ------- | ------------------------------------------------------------ | ----------- |
| WN18RR  | [google drive](https://drive.google.com/drive/folders/1k5mT3d7fldVSSyAYH5KWv3_BI3B2-BXJ?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/1bFmCWfuY1FcGjGF26UHZrg) `axo7`   |
|FB15k-237| [google drive](https://drive.google.com/drive/folders/1aNkuAIQeFOfN4B04xnBOzxhMZKNMxoBH?usp=sharing) |   [baidu drive](https://pan.baidu.com/s/1DK0abYqfvtAPamlLULX4BQ)  `ju9t`            |
| MetaQA  | [google drive](https://drive.google.com/drive/folders/1q4kph9nd4ADjvkPIZvAwYbqza7o7DFt9?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/1AYnfjDAM7z8G3QeEqHTKhg) `hzc9`           |
| KG20C   | [google drive](https://drive.google.com/drive/folders/1AJCRYiNJUFc7inwvrvarwK2hbZEyDALE?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/18pqe1Zz2iT9vY7v5YadUSQ)  `stnh`           |
| CSKB  | [google drive](https://drive.google.com/drive/folders/1A8ofqSOw2xuCkfZgZJkZiTIf-pNG45yi?usp=share_link)  | [baidu drive](https://pan.baidu.com/s/11gleLSKhxN5dIJ3opa9h6g?pwd=endu) `endu`






### Run your first experiment

We provide four tasks in our toolkit as Knowledgeg Graph Completion (KGC), Question Answering (QA), Recomandation (REC) and LAnguage Model Analysis (LAMA).

* `KGC` is our basic task to the knowledge graph embedding and evaluate the ability of the models.
** You can run the script under `kgc` folder to train the model and get the KG embeddings (take `simkgc` as example).
    ```shell
    bash ./scripts/kgc/simkgc.sh
    ```
    
* For `QA` task, you can run the script files under `metaqa`.
** We suggest you use generative model to solve the `QA` task as below:
    ```shell
    bash ./scripts/metaqa/run.sh
    ```
* For `REC` task, you need to firstly get the KG embeddings and then train the rec system models.
** use two-stage scripts below:
    ```shell
    bash ./scripts/kgrec/pretrain_item.sh
    bash ./scripts/kgrec/ml20m.sh
    ```

*  For `LAMA` task, you can use the files under `lama`.
**  We provide `BERT` and `RoBERTa` PLMs to evaluate their performance and with our KG embeddings (plet).
    ```shell
    bash ./scripts/lama/lama_roberta.sh
    ```


### Implemented Models
| Models |Knowledge Graph Completion|Question Answering|Recomandation|LAnguage Model Analysis|
| -----------: | :------: | :------: | :------: |:------: |
| KG-BERT |  ✔  |  ✔  | | |
| GenKGC |  ✔  |      | | |
|   KGT5 |  ✔   |  ✔  | | | 
| kNN-KGE |  ✔   |    | ✔  |✔ |
| SimKGC |  ✔   |      | | | 



### Process on your own data

For each knowledge graph, we have 5 files.
* `train.tsv`, `dev.tsv`, `test.tsv`, list as (h, r, t) for entity id and relation id (start from 0).
* `entity2text.txt`, as (entity_id, entity description).
* `relation2text.txt` , as (relation_id, relation description).

### For downstream tasks

<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/inspire-1.png" width="750px">
</div>

### Using Large Langugae Models (LLMs)

* use text-davince-003 for KGC (Link Prediction)

Before running, please check that you already have the `MidRes.json` available in dataset, or you may run `python ./LLM/create_midres.py` to generate it.

And modify the `api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"` in `./LLM/gpt3kgc.py` with your own openai.api_key.

Once it is available, you can proceed to run the code under the `LLM` folder.
```
python ./LLM/gpt3kgc.py
```

* use text-davince-001/002/003 for commonsense reasoning

Before running, please check that you already have downloaded the ATOMIC2020 dataset to `dataset/atomic_2020_data` and have the file `dataset/atomic_2020_data/test.jsonl` available in dataset, or you may run `python LLM/atomic2020_process.py` to generate it.

And modify the `api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"` in `./LLM/atomic2020_res.py` with your own openai.api_key.

Once it is available, you can proceed to run the following code:
```
python LLM/atomic2020_res.py
```

After running, result file `test_result.json` is available under fold `dataset/atomic_2020_data`.

### Contact Information

For help or issues using the models, please submit a GitHub issue.

# Citation
If you use the code, please cite the following paper:


```bibtex
@article{DBLP:journals/corr/abs-2210-00305,
  author    = {Xin Xie and
               Zhoubo Li and
               Xiaohan Wang and
               Shumin Deng and
               Feiyu Xiong and
               Huajun Chen and
               Ningyu Zhang},
  title     = {PromptKG: {A} Prompt Learning Framework for Knowledge Graph Representation
               Learning and Application},
  journal   = {CoRR},
  volume    = {abs/2210.00305},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2210.00305},
  doi       = {10.48550/arXiv.2210.00305},
  eprinttype = {arXiv},
  eprint    = {2210.00305},
  timestamp = {Fri, 07 Oct 2022 15:24:59 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2210-00305.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Other KG Representation Open-Source Projects

- [OpenKE](https://github.com/thunlp/OpenKE)
- [LibKGE](https://github.com/uma-pi1/kge)
- [CogKGE](https://github.com/jinzhuoran/CogKGE)
- [PyKEEN](https://github.com/pykeen/pykeen)
- [GraphVite](https://graphvite.io/)
- [Pykg2vec](https://github.com/Sujit-O/pykg2vec)
- [PyG](https://github.com/pyg-team/pytorch_geometric)
- [CogDL](https://github.com/THUDM/cogdl)
- [NeuralKG](https://github.com/zjukg/NeuralKG)
- [KGxBoard](https://github.com/neulab/KGxBoard)
