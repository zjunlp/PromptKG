<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/logo.svg" width="350px">
    <p> <b>
        A Prompt Learning Framework for Knowledge Graph  with Pre-trained Language Models.</b>
    </p>
</div>



------



# Implemented Models



| Models | KGC  |  QA  |
| -----: | :--: | :--: |
| GenKGC |  √   |      |
|   KGT5 |  √   |  √   |
| KnnKGE |  √   |      |
| SimKGC |  √   |      |




# Quick Start

## Installation

### **Step1**

Create the conda env.

```shell
conda create -n promptkg python=3.8
```

### **Step2**

Install the dependence.

```shell
pip install -r requirements.txt
```

### Step3

Install our preprocessed datasets and put them into the `dataset` folder.

| Dataset | Google Drive                                                 | Baidu Cloud |
| ------- | ------------------------------------------------------------ | ----------- |
| WN18RR  | [google drive](https://drive.google.com/drive/folders/1k5mT3d7fldVSSyAYH5KWv3_BI3B2-BXJ?usp=sharing) |             |
| MetaQA  | [google drive](https://drive.google.com/drive/folders/1q4kph9nd4ADjvkPIZvAwYbqza7o7DFt9?usp=sharing) |             |





## Run your first experiment

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

## Framework

<div align="center">
    <img src="https://github.com/zjunlp/PromptKG/blob/main/resources/framework.jpg" width="550px">
</div>



## TODO 

- [ ] add essemble-model for using
- [ ] add more kgc models based on pretrained language models
