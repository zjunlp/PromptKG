
<h1 align="center">
    <p>An Open Source Library for Downstream tasks of Knowledge Graphs Based on Pre-trained Language Models</p>
</h1>


# Implemented Models


# Quick Start

## Installation

### **Step1**

Create the conda env.

```shell
conda create -n promptkg python=3.8
```

### **Step2**

Install the denpency.

```shell
pip install -r requirements.txt
```

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


