<div align="center">

<img src="https://github.com/zjunlp/PromptKG/blob/main/resources/logo.svg" width="350px">


**A Prompt Learning Framework for Knowledge Graph Representation Learning and Application**

</div>

**A Prompt Learning Framework for Knowledge Graph  with Pre-trained Language Models  across ```tasks``` (predictive and generative), and ```modalities``` (language, image, vision + language, etc.)**

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

| Dataset | Google Drive                                                 | Baidu Cloud |
| ------- | ------------------------------------------------------------ | ----------- |
| WN18RR  | [google drive](https://drive.google.com/drive/folders/1k5mT3d7fldVSSyAYH5KWv3_BI3B2-BXJ?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/1bFmCWfuY1FcGjGF26UHZrg) `axo7`   |
|FB15k-237| [google drive](https://drive.google.com/drive/folders/1aNkuAIQeFOfN4B04xnBOzxhMZKNMxoBH?usp=sharing) |   [baidu drive](https://pan.baidu.com/s/1DK0abYqfvtAPamlLULX4BQ)  `ju9t`            |
| MetaQA  | [google drive](https://drive.google.com/drive/folders/1q4kph9nd4ADjvkPIZvAwYbqza7o7DFt9?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/1AYnfjDAM7z8G3QeEqHTKhg) `hzc9`           |
| KG20C   | [google drive](https://drive.google.com/drive/folders/1AJCRYiNJUFc7inwvrvarwK2hbZEyDALE?usp=sharing) |  [baidu drive](https://pan.baidu.com/s/18pqe1Zz2iT9vY7v5YadUSQ)  `stnh`           |






### Run your first experiment

We first provide `link-prediction` as our example downstream task.
For `link-prediction` task, we provide `FB15k-237` and `WN18RR` dataset for use.
You can use the command below to run the `simkgc` model on `FB15k-237` dataset.

```shell
bash ./scripts/simkgc/fb15k.sh
```

For `KNN-KGE`, you can switch to the `knn-kge` scripts, and run the command below

```shell
bash ./scripts/knnkge/fb15k.sh
```


### Implemented Models
| Models | KGC  |  QA  |  LAMA |
| -----------: | :------: | :------: |:------: |
| KG-BERT |  ✔  |  ✔  | |
| GenKGC |  ✔  |      | |
|   KGT5 |  ✔   |  ✔  | |
| kNN-KGE |  ✔   |      |✔ |
| SimKGC |  ✔   |      | |

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
