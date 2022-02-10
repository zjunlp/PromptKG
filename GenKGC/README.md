# KNN-KG
Code for our paper "[From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer](https://arxiv.org/abs/2202.02113)"

Requirements
==========
To install requirements:

```
pip install -r requirements.txt
```


Run the experiments
==========

## Training & Inference

Use the script in the `scripts` fold. Take `fb15k-237` dataset for example.

```shell
./scripts/fb15k.sh
```



# Citation
If you use the code, please cite the following paper:

```bibtex
@article{DBLP:journals/corr/abs-2202-02113,
  author    = {Xin Xie and
               Ningyu Zhang and
               Zhoubo Li and
               Shumin Deng and
               Hui Chen and
               Feiyu Xiong and
               Mosha Chen and
               Huajun Chen},
  title     = {From Discrimination to Generation: Knowledge Graph Completion with
               Generative Transformer},
  journal   = {CoRR},
  volume    = {abs/2202.02113},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.02113},
  eprinttype = {arXiv},
  eprint    = {2202.02113},
  timestamp = {Wed, 09 Feb 2022 15:43:35 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-02113.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
