[**中文说明**](https://github.com/zjunlp/PromptKG/tree/main/research/PromptKGC/README_CN.md) | [**English**](https://github.com/zjunlp/PromptKG/tree/main/research/PromptKGC/README.md)


# PromptKGC

This project is mainly about the dataset and code of the prompt-based few-shot knowledge graph completion.

## dataset

We have placed under `dataset` the already processed dataset of low-sample knowledge graphs for scholars to use.

## Dependencies
Use ``pip`` to download the dependency libraries needed for the code

```shell
pip install -r requirements.txt
```

## Run

Using the `FB15k237` dataset as an example, we just need to run the following script after installing the corresponding dependency libraries.
And we prepare 4 dataset in our `dataset` folder for use.

```shell
bash . /scripts/fb15.sh
bash . /scripts/fb15_few.sh # for few shot fb15k-237
bash . /scripts/wn18rr.sh # for wn18rr 
bash . /scripts/umls.sh # for umls
```
## Citation

```bibtex
@Article{zhang2022promptkgc,
title = {Knowledge Collaborative Fine-tuning for Low-resource Knowledge Graph Completion},
author = {Ningyu Zhang and Xin Xie and Xiang Chen and Deng Shumin and Ye Hongbin and Chen Huajun},
 journal = {Journal of Software},
 volume = {33},
 number = {10},
 year = {2022}
}
```
