# GenKGC
Code for our paper "[From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer](https://arxiv.org/pdf/2202.02113.pdf)"

We recommand to run our models in `PromptKG/toolkit`, we will continue to update all features in that repo.

## Requirements

**Step1** Download the basic code

```bash
git clone --depth 1 https://github.com/zjunlp/PromptKG.git
```

**Step2** Create a virtual environment using `Anaconda` and enter it.<br>


```bash
conda create -n genkgc python=3.8

conda activate genkgc
```
   
**Step3** Enter the task directory

```bash
cd PrompKG/research/GenKGC

pip install -r requirements.txt
```

## Download dataset

[click me.](https://drive.google.com/drive/folders/1carN96-mvbYfW_X1Rt-eLCGjEYx3iOda?usp=sharing)

You can download [`OpenBG500`](https://github.com/OpenBGBenchmark/OpenBG500) from the link above.

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
@inproceedings{DBLP:conf/www/XieZLDCXCC22,
  author    = {Xin Xie and
               Ningyu Zhang and
               Zhoubo Li and
               Shumin Deng and
               Hui Chen and
               Feiyu Xiong and
               Mosha Chen and
               Huajun Chen},
  editor    = {Fr{\'{e}}d{\'{e}}rique Laforest and
               Rapha{\"{e}}l Troncy and
               Elena Simperl and
               Deepak Agarwal and
               Aristides Gionis and
               Ivan Herman and
               Lionel M{\'{e}}dini},
  title     = {From Discrimination to Generation: Knowledge Graph Completion with
               Generative Transformer},
  booktitle = {Companion of The Web Conference 2022, Virtual Event / Lyon, France,
               April 25 - 29, 2022},
  pages     = {162--165},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3487553.3524238},
  doi       = {10.1145/3487553.3524238},
  timestamp = {Thu, 18 Aug 2022 10:55:04 +0200},
  biburl    = {https://dblp.org/rec/conf/www/XieZLDCXCC22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
