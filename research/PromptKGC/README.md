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
```shell
bash . /scripts/fb15.sh
```

## Citation
If you are interested in our work or have used our code in your project please reference it in the following format.


```bibtex

@Article{20220,
title = {基于知识协同微调的低资源知识图谱补全方法},
author = {张宁豫,谢辛,陈想,邓淑敏,叶宏彬,陈华钧},
 journal = {软件学报},
 volume = {33},
 number = {10},
 pages = {0},
 numpages = {16.0000},
 year = {2022},
 month = {},
 doi = {10.13328/j.cnki.jos.006628},
 publisher = {科学出版社}
}
```
