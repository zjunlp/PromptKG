<div align="center">

<img src="https://github.com/zjunlp/PromptKG/blob/main/resources/logo.svg" width="350px">


 
</div>

 **A Gallery of Prompt Learning & KG-related research works, toolkits, and paper-list.** 


| Directory | Description |
|-----------|-------------|
| [Research](Research) | • A collection of prompt learning-related **research model implementations** |
| [LambdaKG](LambdaKG) | • A **library and benchmark** for PLM-based KG embeddings |
| [DeltaKG](DeltaKG) | • A **library and benchmark** for dynamically editing and visualizing PLM-based KG embeddings |
| [Tutorial-notebooks](Tutorial-notebooks) | • **Tutorial notebooks** for beginners |

# Tutorials

- Zero- and Few-Shot NLP with Pretrained Language Models. AACL 2022 Tutorial  \[[ppt](https://github.com/allenai/acl2022-zerofewshot-tutorial)\] 
- Data-Efficient Knowledge Graph Construction. CCKS2022 Tutorial  \[[ppt](https://drive.google.com/drive/folders/1xqeREw3dSiw-Y1rxLDx77r0hGUvHnuuE)\] 
- Efficient and Robuts Knowledge Graph Construction. AACL-IJCNLP Tutorial  \[[ppt](https://github.com/NLP-Tutorials/AACL-IJCNLP2022-KGC-Tutorial)\] 
- Knowledge Informed Prompt Learning. MLNLP 2022 Tutorial (Chinese) \[[ppt](https://person.zju.edu.cn/person/attachments/2022-11/01-1668830598-859129.pdf)\] 

# Surveys

* Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models  (on arxiv 2021) \[[paper](https://arxiv.org/abs/2203.06904)\]
* Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing  (ACM Computing Surveys 2021) \[[paper](https://arxiv.org/abs/2107.13586)\]
* reStructured Pre-training (on arxiv 2022) \[[paper](https://arxiv.org/abs/2206.11147)\]
* A Review on Language Models as Knowledge Bases  (on arxiv 2022) \[[paper](https://arxiv.org/abs/2204.06031)\]

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

*Multimodal*

- Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction, in NAACL 2022 (Findings). [\[pdf\]](https://arxiv.org/pdf/2205.03521.pdf)
- Visual Prompt Tuning, in ECCV 2022. [\[pdf\]](https://arxiv.org/abs/2203.12119)
- CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models, in EMNLP 2022. [\[pdf\]](https://arxiv.org/abs/2109.11797)
- Learning to Prompt for Vision-Language Models, in IJCV 2022. [\[pdf\]](https://arxiv.org/abs/2109.01134)
- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models, in NeurIPS 2022.  [\[pdf\]](https://arxiv.org/abs/2209.07511) 

*Advanced Tasks*
- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5), in ACM RecSys 2022. [\[pdf\]](https://arxiv.org/abs/2203.13366) 
- Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning, in KDD 2022.  [\[pdf\]](https://arxiv.org/abs/2206.09363) 
- PromptEM: Prompt-tuning for Low-resource Generalized Entity Matching, in Arxiv 2022. [\[pdf\]](https://arxiv.org/abs/2207.04802) 
- VIMA: General Robot Manipulation with Multimodal Prompts, in Arxiv 2022. [\[pdf\]](https://arxiv.org/abs/2210.03094)
- Unbiasing Retrosynthesis Language Models with Disconnection Prompts, in Arxiv 2022. [\[pdf\]](https://chemrxiv.org/engage/chemrxiv/article-details/6328d0b8ba8a6d04fc551df7)

## Prompt (PLMs) for Knowledge

*Knowledge Probing*

- How Much Knowledge Can You Pack Into the Parameters of a Language Model? in EMNLP 2020.  [\[pdf\]](https://aclanthology.org/2020.emnlp-main.437/)
- Prompting as Probing: Using Language Models for Knowledge Base Construction, in  [\[pdf\]](https://arxiv.org/abs/2208.11057)
- Language Models As or For Knowledge Bases
  *Simon Razniewski, Andrew Yates, Nora Kassner, Gerhard Weikum.   [\[pdf\]](https://arxiv.org/abs/2110.04888)
- Materialized Knowledge Bases from Commonsense Transformers.** CSRR2022  
  *Tuan-Phong Nguyen, Simon Razniewski.*[[pdf](https://aclanthology.org/2022.csrr-1.5/)]  
- Do Language Models Learn Position-Role Mappings?   https://arxiv.org/abs/2202.03611)],2022.2    
- Time-Aware Language Models as Temporal Knowledge Bases.** TACL2022  
  *Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, William W. Cohen.*[[pdf](https://aclanthology.org/2022.tacl-1.15/)]  
- Can Generative Pre-trained Language Models Serve as Knowledge Bases for Closed-book QA?** ACL2021  
  *Cunxiang Wang, Pai Liu, Yue Zhang.*[[pdf](https://aclanthology.org/2021.acl-long.251/)]  
- Language models as knowledge bases: On entity representations, storage capacity, and paraphrased queries, in EACL2021. [\[pdf\]](https://aclanthology.org/2021.eacl-main.153/)
- Scientific language models for biomedical knowledge base completion: an empirical study.**  
  *Rahul Nadkarni, David Wadden, Iz Beltagy, Noah A. Smith, Hannaneh Hajishirzi, Tom Hope.*[[pdf](https://arxiv.org/abs/2106.09700)]  
- Multilingual LAMA: Investigating knowledge in multilingual pretrained language models.** EACL2021  
  *Nora Kassner, Philipp Dufter, Hinrich Schütze.*[[pdf](https://aclanthology.org/2021.eacl-main.284/)]  
- Pre-trained language models as knowledge bases for Automotive Complaint Analysis.**  
  *V. D. Viellieber, M. Aßenmacher.*[[pdf](https://arxiv.org/abs/2012.02558)],2020.12  
- How Can We Know What Language Models Know ?** TACL2020  
  *Zhengbao Jiang, Frank F. Xu, Jun Araki, Graham Neubig.*[[pdf](https://aclanthology.org/2020.tacl-1.28/)]  
- How Context Affects Language Models' Factual Predictions.**  
  *Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel.*[[pdf]
  
*Knowledge Graph Embedding (We provide a library and benchmark [lambdaKG](https://github.com/zjunlp/PromptKG/tree/main/lambdaKG))*

- KG-BERT: BERT for knowledge graph completion

- StATIK: Structure and Text for Inductive Knowledge Graph

- Joint Language Semantic and Structure Embedding for Knowledge Graph Completion

- Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach

- Language Models as Knowledge Embeddings

- From Discrimination to Generation: Knowledge Graph Completion with Generative Transformer

- Reasoning Through Memorization: Nearest Neighbor Knowledge Graph Embeddings
 
- SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models, in ACL 2022.  [\[pdf\]](https://arxiv.org/abs/2203.02167)

- Sequence to Sequence Knowledge Graph Completion and Question Answering

- Structure-Augmented Text Representation Learning for Efficient Knowledge Graph Completion

*Analysis*

- Knowledgeable or Educated Guess? Revisiting Language Models as Knowledge Bases, in ACL 2022
- Emergent Abilities of Large Language Models
- Knowledge Neurons in Pretrained Transformers, in ACL 2022. [\[pdf\]]()
- Finding Skill Neurons in Pre-trained Transformer-based Language Models
- Do Prompts Solve NLP Tasks Using Natural Languages? 
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

## Contact Information

For help or issues using the tookits, please submit a GitHub issue.
