import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_glove_embeddings(path, word2idx, embedding_dim):
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype="float32")
                embeddings[index][:len(vector)] = vector
        return torch.from_numpy(embeddings).float()


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_dict,
        glove_file=None,
        emb_dim=104,
        dropout_p=0.1,
        word_embed_dim=50,
    ):
        super(TextCNN, self).__init__()
        # TextCNN from HiLAP
        Ks = [3, 4, 5]
        Ci = 1
        Co = 1000
        self.embed = nn.Embedding(len(vocab_dict), word_embed_dim)
        if glove_file:
            embeddings = load_glove_embeddings(
                glove_file, vocab_dict, embedding_dim=word_embed_dim
            )
            self.embed.weight = nn.Parameter(embeddings)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(Ci, Co, (K, word_embed_dim)) for K in Ks]
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(len(Ks) * Co, emb_dim)

    def forward(self, x):
        """
        Inputs:
            `x` is a list of tokenized documents as token ids
        Outputs:
            Embedding of `x`
        """
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        return x

class LabelEmbedModel(nn.Module):
    def __init__(self, n_labels, emb_dim=104, dropout_p=0.4, eye=False):
        super(LabelEmbedModel, self).__init__()
        self.eye = eye
        self.dropout = nn.Dropout(dropout_p)
        self.e = nn.Embedding(
                    n_labels, emb_dim,
                    max_norm=1.0,
                    # sparse=True,
                    scale_grad_by_freq=False
                )
        self.init_weights()

    def init_weights(self, scale=1e-4):
        if self.eye:
            torch.nn.init.eye_(self.e.weight)
        else:
            self.e.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, x):
        """
        x is the list of labels eg [1,3,8]
        """
        return self.dropout(self.e(x))
