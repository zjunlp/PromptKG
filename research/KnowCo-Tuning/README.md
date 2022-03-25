# KnowCo-Tuning

本项目主要是关于[基于知识协同微调的低资源知识图谱补全方法](http://jos.org.cn/jos/article/abstract/6628?st=search)的数据集和代码。

## 数据集

我们在`dataset`下面放置了已经处理好的少样本知识图谱数据集供学者使用。

## 依赖项
使用`pip`来下载代码所需要的依赖库

```shell
pip install -r requirements.txt
```

## 运行

以`FB15k237`数据集为例子，我们只需要在安装相对应依赖库之后运行以下脚本即可。
```shell
bash ./scripts/fb15.sh
```

## 引用
如果您对我们工作感兴趣或在项目中使用了我们的代码请以如下格式引用。

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
