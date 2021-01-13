## Joint Learning of Hyperbolic Label Embeddings for Hierarchical Multi-label Classification

> [Joint Learning of Hyperbolic Label Embeddings for Hierarchical Multi-label Classification]()  
> Soumya Chatterjee, Ayush Maheshwari, Ganesh Ramakrishnan and Saketha Nath Jagaralpudi  
> To Appear at European Chapter of the Association for Computational Linguistics (__EACL__) 2021

### Requirements

`environment.yml` has the depedencies

Download `glove.6B.300d.txt` from [here](https://nlp.stanford.edu/projects/glove/) in `GloVe` folder

Please refer to [HiLAP](https://github.com/morningmoni/HiLAP) for the dataset instructions. Put the required dataset files in folders `rcv1`, `yelp` or `nyt` and run `data_utils/gen_json_<dataset>.py` for preprocessing the data.

### Run

Run `main.py` using the arguments `--exp_name`<br>
&emsp;`--flat` for `Model_flt`<br>
&emsp;`--cascaded_step1` and `--cascaded_step2` for `Model_cas`<br> 
&emsp;`--joint` for `Model_jnt`

Specify the dataset using `--dataset` and the base text classification model using `--base_model`.

For examples, please refer `Synthetic/all_expts.sh`.

### Acknowledgement

- [HiLAP](https://github.com/morningmoni/HiLAP) for data processing and TextCNN model
- [Poincare Embeddings](https://github.com/facebookresearch/poincare-embeddings) for Poincare utils
- [NeuralNLP-NeuralClassifier](https://github.com/Tencent/NeuralNLP-NeuralClassifier) for TextRCNN model
- `bert-base-uncased-vocab.txt` is from [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)

### Citation:

```
Coming Soon
```
