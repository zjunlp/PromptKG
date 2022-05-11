[**中文说明**](https://github.com/zjunlp/PromptKG/tree/main/research/PromptKGC/README_CN.md) | [**English**](https://github.com/zjunlp/PromptKG/tree/main/research/PromptKGC/README.md)

# PromptKGC

本项目主要是关于基于提示学习的少样本知识图谱补全的数据集和代码。

## 数据集

我们在`dataset`下面放置了已经处理好的少样本知识图谱数据集供学者使用。

## 依赖项
使用`pip`来下载代码所需要的依赖库

```shell
pip install -r requirements.txt
```

## 运行

以`FB15k237`数据集为例子，我们只需要在安装相对应依赖库之后运行以下脚本即可。
我们准备了4个数据集供使用和训练。

```shell
bash . /scripts/fb15.sh
bash . /scripts/fb15_few.sh # for few shot fb15k-237
bash . /scripts/wn18rr.sh # for wn18rr 
bash . /scripts/umls.sh # for umls
```

## 引用

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
