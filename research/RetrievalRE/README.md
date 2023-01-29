# RetrieveRE

Code for the SIGIR 2022 short paper "[Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt Tuning](https://arxiv.org/pdf/2205.02355.pdf)"

# Related Work

## RetroPrompt (NeurIPS2022)
* Our method is related to [Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning](https://arxiv.org/abs/2205.14704) that has been accepted by NeurIPS2022. The project address is visible at [RetroPrompt](https://github.com/zjunlp/PromptKG/tree/main/research/RetroPrompt).

## KnowPrompt (WWW2022)
* Our method is based on [KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction](https://dl.acm.org/doi/abs/10.1145/3485447.3511998) that has been accepted by WWW2022. The project address is visible at [KnowPrompt](https://github.com/zjunlp/KnowPrompt).

## PTR
* Our paper is also related to [PTR: Prompt Tuning with Rules for Text Classification](https://arxiv.org/abs/2105.11259).


# Model Architecture

<div align=center>
<img src="resource/model.png" width="75%" height="75%" />
</div>

The inference procedure of our RetrievalRE.


## Requirements

**Step1** Download the basic code

```bash
git clone --depth 1 https://github.com/zjunlp/PromptKG.git
```

**Step2** Create a virtual environment using `Anaconda` and enter it.<br>


```bash
conda create -n retrievalre python=3.8

conda activate retrievalre
```
   
**Step3** Enter the task directory

```bash
cd PrompKG/research/RetrievalRE

pip install -r requirements.txt
```


# Datasets

To evaluate RetrieveRE, we conduct experiments on 5 RE datasets, include SEMEVAL, DialogRE, TACRED-Revisit, Re-TACRE and TACRED. You can download them via this [link](https://drive.google.com/file/d/1BQW3ekxJ96w2652FndRtcEIHp6Gwy_py/view?usp=sharing).


The expected structure of files is:


```
RetrieveRE
 |-- dataset
 |    |-- semeval
 |    |    |-- train.txt       
 |    |    |-- dev.txt
 |    |    |-- test.txt
 |    |    |-- temp.txt
 |    |    |-- rel2id.json
 |    |-- dialogue
 |    |    |-- train.json       
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- rel2id.json
 |    |-- tacred
 |    |    |-- ...
 |    |-- tacrev
 |    |    |-- ...
 |    |-- retacred
 |    |    |-- ...
 |-- scripts
 |    |-- knn_scripts
 |    |    |-- get_knn_semeval.sh
 |    |    |-- combine_knn_semeval.sh
 |    |-- semeval.sh
 |    |-- dialogue.sh
 |    |-- ...
 |-- data
 |-- lit_models
 |-- models
 |-- get_label_word.py
 |-- main.py
```


# How to run

## Initialize the answer words

Use the comand below to get the answer words to use in the training.

```shell
python get_label_word.py --model_name_or_path bert-large-uncased  --dataset_name semeval
```

The `{answer_words}.pt`will be saved in the dataset, you need to assign the `model_name_or_path` and `dataset_name` in the `get_label_word.py`.

## Split few-shot dataset

Download the data first, and put it to `dataset` folder. Run the comand below, and get the few shot dataset.

```shell
python generate_k_shot.py --data_dir ./dataset --k 8 --dataset semeval
cd dataset
cd semeval
cp rel2id.json val.txt test.txt ./k-shot/8-1
```
You need to modify the `k` and `dataset` to assign k-shot and dataset. Here we default seed as 1,2,3,4,5 to split each k-shot, you can revise it in the `generate_k_shot.py`

## Run

There are three steps to train RetrieveRE model:

- Train the re model with robert-base.
- Use the trained re model to get knn data.
- Inference with the trained re model and knn data.

#### Example for SEMEVAL
Train the Retrieve model on SEMEVAL with the following command:

```bash
bash scripts/semeval.sh 
bash scripts/knn_scripts/get_knn_semeval.sh
bash scripts/knn_scripts/combine_knn_semeval.sh
```
As the scripts  for `TACRED-Revist`, `Re-TACRED`, `Wiki80` included in our paper are also provided, you just need to run it like above example.

#### Example for DialogRE
As the data format of DialogRE is very different from other dataset, Class of processor is also different. 
Train the RetieveRE model on DialogRE with the following command:

```bash
bash scripts/dialogue.sh 
bash scripts/knn_scripts/get_knn_dialogue.sh
bash scripts/knn_scripts/combine_knn_dialogue.sh
```

# Papers for the Project & How to Cite
If you use or extend our work, please cite the paper as follows:

```bibtex
@inproceedings{DBLP:conf/sigir/ChenLZTHSC22,
  author    = {Xiang Chen and
               Lei Li and
               Ningyu Zhang and
               Chuanqi Tan and
               Fei Huang and
               Luo Si and
               Huajun Chen},
  editor    = {Enrique Amig{\'{o}} and
               Pablo Castells and
               Julio Gonzalo and
               Ben Carterette and
               J. Shane Culpepper and
               Gabriella Kazai},
  title     = {Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt
               Tuning},
  booktitle = {{SIGIR} '22: The 45th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Madrid, Spain, July 11 -
               15, 2022},
  pages     = {2443--2448},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3477495.3531746},
  doi       = {10.1145/3477495.3531746},
  timestamp = {Mon, 11 Jul 2022 12:19:25 +0200},
  biburl    = {https://dblp.org/rec/conf/sigir/ChenLZTHSC22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
