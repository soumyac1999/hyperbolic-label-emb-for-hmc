import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
import torch.nn.functional as F
import pickle
import numpy as np


class Labels:
    def __init__(self, label_file):
        self.labels = [l.strip() for l in open(label_file)]
        self.stoi = {l: i for i, l in enumerate(self.labels)}
        self.n_labels = len(self.labels)

    def multihot(self, x):
        """
        Given a list of labels as strings (for a document),
        return the corresponding multi-hot vector
        """
        with torch.no_grad():
            ret = torch.zeros(self.n_labels)
            for l in x:
                ret += nn.functional.one_hot(torch.tensor(self.stoi[l]), self.n_labels)
        return ret

    def to_indices(self, multihot):
        return torch.arange(self.n_labels)[multihot]


class LabelDataset(Dataset):
    def __init__(self, N, nnegs=5):
        super(LabelDataset, self).__init__()
        self.items = N.shape[0]
        # TODO: Check if N[i. j] != 0
        self.len = N.shape[0]*N.shape[0]
        self.N = N
        self.nnegs = nnegs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration
        t = idx//self.items
        h = idx%self.items
        negs = np.arange(self.items)[self.N[t][h] == 1.0]
        # Very bad sampling method (TODO: fix)
        negs = negs.repeat(self.nnegs)
        np.random.shuffle(negs)
        return torch.tensor([t, h, *negs[:self.nnegs]])


class TextDataset(Dataset):
    def __init__(self, json_data_file, labels, vocab_dict, n_tokens):
        """
        labels is an object of class Labels()
        """
        WordPiece = BertWordPieceTokenizer(
            "bert-base-uncased-vocab.txt",
            lowercase=True,
            add_special_tokens=False,
            sep_token="",
            cls_token="",
        )
        self.x = []
        self.y = []

        self.similarity = torch.zeros((labels.n_labels, labels.n_labels))

        vocab = set()
        vocab.update(["UNK"])
        for l in tqdm(open(json_data_file)):
            d = json.loads(l)
            WordPieceEncoder = WordPiece.encode(d["text"])
            tokens = WordPieceEncoder.tokens
            self.x.append(tokens)
            vocab.update(tokens)
            self.y.append(labels.multihot(d["label"]))

            li = [labels.stoi[l] for l in d["label"]]
            for i in li:
                for j in li:
                    self.similarity[i, j] += 1
                    self.similarity[j, i] += 1

        self.similarity /= len(self.x)

        self.vocab = {tok: i for i, tok in enumerate(vocab)}

        if vocab_dict != None:
            self.vocab = vocab_dict

        for idx in tqdm(range(len(self.x))):
            self.x[idx] = [
                self.vocab[i] if i in self.vocab else self.vocab["UNK"]
                for i in self.x[idx]
            ][:n_tokens]
            if len(self.x[idx]) < n_tokens:
                self.x[idx] += [self.vocab["UNK"]] * n_tokens
                self.x[idx] = self.x[idx][:n_tokens]

        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration
        return torch.tensor(self.x[idx]), self.y[idx]


class TextLabelDataset(Dataset):
    def __init__(self, json_data_file, label_file, vocab_dict=None, n_tokens=256, nnegs=5, hier_file=None):
        super(TextLabelDataset, self).__init__()

        labels = Labels(label_file)
        self.labels = labels
        self.text_dataset = TextDataset(json_data_file, labels, vocab_dict, n_tokens)
        
        # print(hier_file)
        if hier_file == None:
            similarity_matrix = self.text_dataset.similarity
            n_labels = labels.n_labels
            N = torch.zeros((n_labels, n_labels, n_labels))
            for i in range(n_labels):
                for j in range(n_labels):
                    N[i][j] = 1.0*(similarity_matrix[i, :] <= similarity_matrix[i][j])
        else:
            n_labels = labels.n_labels
            edges = torch.zeros((n_labels, n_labels))
            for l in open(hier_file):
                ls = l.split('\t')
                h = ls[0]
                t = ls[1:]
                h = labels.stoi[h.strip()]
                t = [labels.stoi[v.strip()] for v in t]
                edges[h, t] = 1
            for k in range(n_labels):
                for i in range(n_labels):
                    for j in range(n_labels):
                        edges[i][j] = edges[i][j] + edges[i][k]*edges[k][j]
            N = torch.zeros((n_labels, n_labels, n_labels))
            for i in range(n_labels):
                for j in range(n_labels):
                    N[i][j] = 1 - edges[i]

        self.label_dataset = LabelDataset(N)

        self.n_labels = n_labels        
        self.len = max(len(self.text_dataset), len(self.label_dataset))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return *self.text_dataset[idx%len(self.text_dataset)], self.label_dataset[idx%len(self.label_dataset)]

