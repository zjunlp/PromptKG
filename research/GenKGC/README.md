# GenKGC
Code for our paper "[From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer](https://arxiv.org/abs/2202.02113)"

Requirements
==========
To install requirements:

```
pip install -r requirements.txt
```

## Download dataset

[click me.](https://drive.google.com/drive/folders/1carN96-mvbYfW_X1Rt-eLCGjEYx3iOda?usp=sharing)

You can download `OpenBG500` from the link above.

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
@article{xie2022discrimination,
  title={From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer},
  author={Xie, Xin and Zhang, Ningyu and Li, Zhoubo and Deng, Shumin and Chen, Hui and Xiong, Feiyu and Chen, Mosha and Chen, Huajun},
  journal={arXiv preprint arXiv:2202.02113},
  year={2022}
}
```
