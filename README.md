<div align="center">

<img src="https://github.com/zjunlp/PromptKG/blob/main/resources/logo.svg" width="350px">


**A Prompt Learning Framework for Knowledge Graph Representation Learning and Application**

</div>

**A Prompt Learning Model Zoo with Pre-trained Language Models  across ```tasks``` (predictive and generative), and ```modalities``` (language, image, vision + language, etc.)**

Paper:
[PromptKG: A Prompt Learning Framework for Knowledge Graph Representation Learning and Application](https://arxiv.org/abs/2210.00305)


| Directory | Description |
|-----------|-------------|
| [toolkit](toolkit) | • A prompt-based toolkit for knowledge graph construction and application |
| [research](research) | • A collection of research model implementations in Pytorch by researchers |
| [tutorial-notebooks](tutorial-notebooks) | • Tutorial notebooks for beginners |


## Toolkit


## Quick Start

### Installation

#### **Step1**

Create the conda env.

```shell
conda create -n promptkg python=3.8
```

#### **Step2**

Install the dependence.

```shell
pip install -r requirements.txt
```

#### Step3

Install our preprocessed datasets and put them into the `dataset` folder.

| Dataset (KGC) | Google Drive                                                 | Baidu Cloud |
| ------- | ------------------------------------------------------------ | ----------- |
| WN18RR  | [google drive](https://drive.google.com/drive/folders/1k5mT3d7fldVSSyAYH5KWv3_BI3B2-BXJ?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/1bFmCWfuY1FcGjGF26UHZrg) `axo7`   |
|FB15k-237| [google drive](https://drive.google.com/drive/folders/1aNkuAIQeFOfN4B04xnBOzxhMZKNMxoBH?usp=sharing) |   [baidu drive](https://pan.baidu.com/s/1DK0abYqfvtAPamlLULX4BQ)  `ju9t`            |
| MetaQA  | [google drive](https://drive.google.com/drive/folders/1q4kph9nd4ADjvkPIZvAwYbqza7o7DFt9?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/1AYnfjDAM7z8G3QeEqHTKhg) `hzc9`           |
| KG20C   | [google drive](https://drive.google.com/drive/folders/1AJCRYiNJUFc7inwvrvarwK2hbZEyDALE?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/18pqe1Zz2iT9vY7v5YadUSQ)  `stnh`           |






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
| Models | &emsp;KGC  &emsp;  | &emsp;QA &emsp;   | &emsp;REC&emsp;   | &emsp;LAMA &emsp;   |
| -----------: | :------: | :------: | :------: |:------: |
| KG-BERT |  ✔  |  ✔  | | |
| GenKGC |  ✔  |      | | |
|   KGT5 |  ✔   |  ✔  | | | 
| kNN-KGE |  ✔   |    | ✔  |✔ |
| SimKGC |  ✔   |      | | | 

### Framework

<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/framework-1.png" width="550px">
</div>

### Process on your own data

For each knowledge graph, we have 5 files.
* `train.tsv`, `dev.tsv`, `test.tsv`, list as (h, r, t) for entity id and relation id (start from 0).
* `entity2text.txt`, as (entity_id, entity description).
* `relation2text.txt` , as (relation_id, relation description).

### For downstream tasks

<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/inspire-1.png" width="350px">
</div>


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

- [OpenKE](https://github.com/jinzhuoran/CogIE)
- [LibKGE](https://github.com/uma-pi1/kge)
- [CogKGE](https://github.com/jinzhuoran/CogKGE)
- [PyKEEN](https://github.com/pykeen/pykeen)
- [GraphVite](https://graphvite.io/)
- [Pykg2vec](https://github.com/Sujit-O/pykg2vec)
- [PyG](https://github.com/pyg-team/pytorch_geometric)
- [CogDL](https://github.com/THUDM/cogdl)
- [NeuralKG](https://github.com/zjukg/NeuralKG)
