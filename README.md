# Joint Learning of Hyperbolic Label Embeddings for Hierarchical Multi-label Classification

> [Joint Learning of Hyperbolic Label Embeddings for Hierarchical Multi-label Classification]()  
> Soumya Chatterjee, Ayush Maheshwari, Saketha Nath Jagaralpudi and Ganesh Ramakrishnan

## Abstract

We consider the problem of multi-label classification, where the labels lie on a hierarchy. However, unlike most existing works in hierarchical multi-label classification, we do not assume that the label-hierarchy is known. Encouraged by the recent success of hyperbolic embeddings in capturing hierarchical relations, we propose to jointly learn the classifier parameters as well as the hyperbolic label embeddings. Such a joint learning is expected to provide a two-fold advantage: i) the classifier generalises better as it leverages the prior knowledge of existence of a hierarchy over the labels, and ii) in addition to the label cooccurrence information, the label-embedding may benefit from the manifold structure of the input datapoints, leading to embeddings that are more faithful to the label hierarchy.

We propose a novel formulation for the joint learning and empirically evaluate its efficacy. The results show that the joint learning improves over the baseline that employs label co-occurrence based pre-trained hyperbolic embeddings. Moreover, the proposed classifiers achieve state-of-the-art generalization on standard benchmarks. We also present evaluation of the hyperbolic embeddings obtained by joint learning and show that they represent the hierarchy more accurately than the other alternatives

## Requirements

`environment.yml` has the depedencies

Download `glove.6B.300d.txt` from [here](https://nlp.stanford.edu/projects/glove/) in `GloVe` folder

Please refer to [HiLAP](https://github.com/morningmoni/HiLAP) for the dataset instructions. Put the required dataset files in folders `rcv1`, `yelp` or `nyt` and run `data_utils/gen_json_<dataset>.py` for preprocessing the data.

## Run

Run `main.py` using the arguments `--exp_name`<br>
&emsp;`--flat` for `Model_flt`<br>
&emsp;`--cascaded_step1` and `--cascaded_step2` for `Model_cas`<br> 
&emsp;`--joint` for `Model_jnt`

Specify the dataset using `--dataset` and the base text classification model using `--base_model`.

For examples, please refer `Synthetic/all_expts.sh`.

## Acknowledgement

- [HiLAP](https://github.com/morningmoni/HiLAP) for data processing and TextCNN model
- [Poincare Embeddings](https://github.com/facebookresearch/poincare-embeddings) for Poincare utils
- [NeuralNLP-NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier) for TextRCNN model
- `bert-base-uncased-vocab.txt` is from [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)

## Cite

```
Coming Soon
```
