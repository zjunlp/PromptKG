<div align="center">

<img src="https://github.com/zjunlp/PromptKG/blob/main/resources/logo.svg" width="350px">



 **A Gallery of Prompt Learning & KG-related research works, toolkits, and paper-list.** 
 
  ![](https://img.shields.io/github/last-commit/zjunlp/PromptKG?color=green) ![](https://img.shields.io/badge/PaperNumber-61-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) 
  
</div>

| Directory | Description |
|-----------|-------------|
| [research](research) | • A collection of prompt learning-related **research model implementations** |
| [lambdaKG](lambdaKG) | • A library and benchmark for **PLM-based KG embeddings and applications** |
| [deltaKG](deltaKG) | • A library for **dynamically editing and visualizing PLM-based KG embeddings** |
| [tutorial-notebooks](tutorial-notebooks) | • **Tutorial notebooks** for beginners |

# Table of Contents

* [Tutorials](#Tutorials)
* [Surveys](#Surveys)
* [Papers](#Papers)
   * [Knowledge as Prompt](#Knowledge-as-Prompt)
      * [1. Language Understanding](#Knowledge-as-Prompt)
      * [2. Multimodal](#Knowledge-as-Prompt)
      * [3. Advanced Tasks](#Knowledge-as-Prompt)
   * [Prompt (PLMs) for Knowledge](#prompt-plms-for-knowledge)
      * [1. Knowledge Probing](#prompt-plms-for-knowledge)
      * [2. Knowledge Graph Embedding](#prompt-plms-for-knowledge)
      * [3. Analysis](#prompt-plms-for-knowledge)
* [Contact Information](#Contact-Information)

# Tutorials

- Zero- and Few-Shot NLP with Pretrained Language Models. AACL 2022 Tutorial  \[[ppt](https://github.com/allenai/acl2022-zerofewshot-tutorial)\] 
- Data-Efficient Knowledge Graph Construction. CCKS2022 Tutorial  \[[ppt](https://drive.google.com/drive/folders/1xqeREw3dSiw-Y1rxLDx77r0hGUvHnuuE)\] 
- Efficient and Robuts Knowledge Graph Construction. AACL-IJCNLP Tutorial  \[[ppt](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 
- Knowledge Informed Prompt Learning. MLNLP 2022 Tutorial (Chinese) \[[ppt](https://person.zju.edu.cn/person/attachments/2022-11/01-1668830598-859129.pdf)\] 

# Surveys

* Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models  (on arxiv 2021) \[[paper](https://arxiv.org/abs/2203.06904)\]
* Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing  (ACM Computing Surveys 2021) \[[paper](https://arxiv.org/abs/2107.13586)\]
* reStructured Pre-training (on arxiv 2022) \[[paper](https://arxiv.org/abs/2206.11147)\]
* A Survey of Knowledge-Intensive NLP with Pre-Trained Language Models (on arxiv 2022) \[[paper](https://arxiv.org/abs/2202.08772)\]
* A Survey of Knowledge-Enhanced Pre-trained Language Models  (on arxiv 2022) \[[paper](https://arxiv.org/abs/2211.05994)\]
* A Review on Language Models as Knowledge Bases  (on arxiv 2022) \[[paper](https://arxiv.org/abs/2204.06031)\]
* Generative Knowledge Graph Construction: A Review (EMNLP, 2022) \[[paper](https://arxiv.org/pdf/2210.12714.pdf)\]

# Papers

## Knowledge as Prompt

*Language Understanding*

- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, in NeurIPS 2020.  [\[pdf\]](https://arxiv.org/abs/2005.11401)
- REALM: Retrieval-Augmented Language Model Pre-Training, in ICML 2020.  [\[pdf\]](https://arxiv.org/abs/2002.08909)
- Making Pre-trained Language Models Better Few-shot Learners, in ACL 2022. [\[pdf\]](https://aclanthology.org/2021.acl-long.295/)
- PTR: Prompt Tuning with Rules for Text Classification, in OpenAI 2022. [\[pdf\]](https://arxiv.org/pdf/2105.11259.pdf)
- Label Verbalization and Entailment for Effective Zero- and Few-Shot Relation Extraction, in EMNLP 2021. [\[pdf\]](https://aclanthology.org/2021.emnlp-main.92.pdf)
- RelationPrompt: Leveraging Prompts to Generate Synthetic Data for Zero-Shot Relation Triplet Extraction, in EMNLP 2022 (Findings). [\[pdf\]](https://arxiv.org/abs/2203.09101)
- Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification, in ACL 2022. [\[pdf\]](https://arxiv.org/abs/2108.02035)
- PPT: Pre-trained Prompt Tuning for Few-shot Learning, in ACL 2022. [\[pdf\]](https://arxiv.org/abs/2109.04332)
- Contrastive Demonstration Tuning for Pre-trained Language Models, in EMNLP 2022 (Findings). [\[pdf\]](https://arxiv.org/abs/2204.04392)
- AdaPrompt: Adaptive Model Training for Prompt-based NLP, in arxiv 2022. [\[pdf\]](https://arxiv.org/abs/2202.04824)
- KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction, in WWW 2022. [\[pdf\]](https://arxiv.org/abs/2104.07650)
- Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning, in NeurIPS 2022. [\[pdf\]](https://arxiv.org/abs/2205.14704)
- Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt Tuning, in SIGIR 2022. [\[pdf\]](https://arxiv.org/abs/2205.02355)
- LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting, in COLING 2022. [\[pdf\]](https://aclanthology.org/2022.coling-1.209/)
- Unified Structure Generation for Universal Information Extraction, in ACL 2022. [\[pdf\]](https://aclanthology.org/2022.acl-long.395/)
- LasUIE: Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model, in NeurIPS 2022. [\[pdf\]](https://openreview.net/forum?id=a8qX5RG36jd) 
- Atlas: Few-shot Learning with Retrieval Augmented Language Models, in Arxiv 2022. [\[pdf\]](https://arxiv.org/abs/2208.03299) 
- Don't Prompt, Search! Mining-based Zero-Shot Learning with Language Models, in ACL 2022. [\[pdf\]](https://arxiv.org/abs/2210.14803) 
- Knowledge Prompting in Pre-trained Language Model for Natural Language Understanding, in EMNLP 2022.  [\[pdf\]](https://arxiv.org/abs/2210.08536)
- Unified Knowledge Prompt Pre-training for Customer Service Dialogues, in CIKM 2022.  [\[pdf\]](https://arxiv.org/abs/2208.14652)
- Knowledge Prompting in Pre-trained Language Model for Natural Language Understanding, in EMNLP 2022.  [\[pdf\]](https://arxiv.org/pdf/2210.08536.pdf)

*Multimodal*

- Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction, in NAACL 2022 (Findings). [\[pdf\]](https://arxiv.org/pdf/2205.03521.pdf)
- Visual Prompt Tuning, in ECCV 2022. [\[pdf\]](https://arxiv.org/abs/2203.12119)
- CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models, in EMNLP 2022. [\[pdf\]](https://arxiv.org/abs/2109.11797)
- Learning to Prompt for Vision-Language Models, in IJCV 2022. [\[pdf\]](https://arxiv.org/abs/2109.01134)
- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models, in NeurIPS 2022.  [\[pdf\]](https://arxiv.org/abs/2209.07511) 

*Advanced Tasks*
- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5), in ACM RecSys 2022. [\[pdf\]](https://arxiv.org/abs/2203.13366) 
- Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning, in KDD 2022.  [\[pdf\]](https://arxiv.org/abs/2206.09363) 
- PromptEM: Prompt-tuning for Low-resource Generalized Entity Matching, in VLDB 2023. [\[pdf\]](https://arxiv.org/abs/2207.04802) 
- VIMA: General Robot Manipulation with Multimodal Prompts, in Arxiv 2022. [\[pdf\]](https://arxiv.org/abs/2210.03094)
- Unbiasing Retrosynthesis Language Models with Disconnection Prompts, in Arxiv 2022. [\[pdf\]](https://chemrxiv.org/engage/chemrxiv/article-details/6328d0b8ba8a6d04fc551df7)
- ProgPrompt: Generating Situated Robot Task Plans using Large Language Models, in Arxiv 2022. [\[pdf\]](https://arxiv.org/abs/2209.11302)

## Prompt (PLMs) for Knowledge

*Knowledge Probing*

- How Much Knowledge Can You Pack Into the Parameters of a Language Model? in EMNLP 2020.  [\[pdf\]](https://aclanthology.org/2020.emnlp-main.437/)
- Language Models as Knowledge Bases? in EMNLP 2019. [\[pdf\]](https://aclanthology.org/D19-1250.pdf)
- Materialized Knowledge Bases from Commonsense Transformers, in CSRR 2022. [\[pdf\]](https://aclanthology.org/2022.csrr-1.5/)
- Time-Aware Language Models as Temporal Knowledge Bases, in TACL2022.  [\[pdf\]](https://aclanthology.org/2022.tacl-1.15/)
- Can Generative Pre-trained Language Models Serve as Knowledge Bases for Closed-book QA? in ACL2021.  [\[pdf\]](https://aclanthology.org/2021.acl-long.251/) 
- Language models as knowledge bases: On entity representations, storage capacity, and paraphrased queries, in EACL2021. [\[pdf\]](https://aclanthology.org/2021.eacl-main.153/)
- Scientific language models for biomedical knowledge base completion: an empirical study, in AKBC 2021. [\[pdf\]](https://arxiv.org/abs/2106.09700) 
- Multilingual LAMA: Investigating knowledge in multilingual pretrained language models, in  EACL2021. [\[pdf\]](https://aclanthology.org/2021.eacl-main.284/)
- How Can We Know What Language Models Know ? in TACL 2020. [\[pdf\]](https://aclanthology.org/2020.tacl-1.28/)
- How Context Affects Language Models' Factual Predictions, in AKBC 2020.  [\[pdf\]](https://arxiv.org/abs/2005.04611)
  
*Knowledge Graph Embedding (We provide a library and benchmark [lambdaKG](https://github.com/zjunlp/PromptKG/tree/main/lambdaKG))*

- KG-BERT: BERT for knowledge graph completion, in Arxiv 2020. [\[pdf\]](https://arxiv.org/abs/1909.03193)
- Structure-Augmented Text Representation Learning for Efficient Knowledge Graph Completion, in WWW 2021.  [\[pdf\]](https://arxiv.org/abs/2004.14781)
- StATIK: Structure and Text for Inductive Knowledge Graph, in NAACL 2022. [\[pdf\]](https://aclanthology.org/2022.findings-naacl.46/)
- Joint Language Semantic and Structure Embedding for Knowledge Graph Completion, in COLING.  [\[pdf\]](https://aclanthology.org/2022.coling-1.171/)
- Knowledge Is Flat: A Seq2Seq Generative Framework for Various Knowledge Graph Completion, in COLING.  [\[pdf\]](https://aclanthology.org/2022.coling-1.352/)
- Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach, in ACL 2022.  [\[pdf\]](https://aclanthology.org/2022.findings-acl.282/)
- Language Models as Knowledge Embeddings, in IJCAI 2022.  [\[pdf\]](https://arxiv.org/abs/2206.12617)
- From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer, in WWW 2022.  [\[pdf\]](https://arxiv.org/abs/2202.02113)
- Reasoning Through Memorization: Nearest Neighbor Knowledge Graph Embeddings, in Arxiv 2022.  [\[pdf\]](https://arxiv.org/abs/2201.05575)
- SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models, in ACL 2022.  [\[pdf\]](https://arxiv.org/abs/2203.02167)
- Sequence to Sequence Knowledge Graph Completion and Question Answering, in ACL 2022.  [\[pdf\]](https://arxiv.org/abs/2203.10321)

*Analysis*

- Knowledgeable or Educated Guess? Revisiting Language Models as Knowledge Bases, in ACL 2021. [\[pdf\]](https://aclanthology.org/2021.acl-long.146/)
- Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View, in ACL 2022.  [\[pdf\]](https://arxiv.org/abs/2203.12258)
- How Pre-trained Language Models Capture Factual Knowledge? A Causal-Inspired Analysis, in ACl 2022. [\[pdf\]](https://arxiv.org/abs/2203.16747)
- Emergent Abilities of Large Language Models, in Arxiv 2022.  [\[pdf\]](https://arxiv.org/abs/2206.07682)
- Knowledge Neurons in Pretrained Transformers, in ACL 2022. [\[pdf\]](https://arxiv.org/abs/2104.08696)
- Finding Skill Neurons in Pre-trained Transformer-based Language Models, in EMNLP 2022.  [\[pdf\]](https://arxiv.org/abs/2211.07349)
- Do Prompts Solve NLP Tasks Using Natural Languages? in Arxiv 2022.  [\[pdf\]](https://arxiv.org/abs/2203.00902)
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? in EMNLP 2022. [\[pdf\]](https://arxiv.org/abs/2202.12837)
- Do Prompt-Based Models Really Understand the Meaning of their Prompts? in NAACL 2022.  [\[pdf\]](https://arxiv.org/abs/2109.01247)

## Contact Information

For help or issues using the tookits, please submit a GitHub issue.
