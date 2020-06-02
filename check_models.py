import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
import torch.nn.functional as F
import pickle
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


class TextRCNN(nn.Module):
    def __init__(self, vocab_dict, glove_file, emb_dim, dropout_p=0.5, word_embed_dim=300):
        super(TextRCNN, self).__init__()

        self.embed = nn.Embedding(len(vocab_dict), word_embed_dim)
        if glove_file:
            embeddings = load_glove_embeddings(
                glove_file, vocab_dict, embedding_dim=word_embed_dim
            )
            self.embed.weight = nn.Parameter(embeddings)

        self.rnn = torch.nn.GRU(
            word_embed_dim, 64, num_layers=1, bias=True,
            batch_first=True, bidirectional=True)

        hidden_dimension = 2*64
        self.kernel_sizes = [2,3,4]
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension, 100,
                kernel_size, padding=kernel_size - 1))

        self.top_k = 1
        hidden_size = 3 * 100 * self.top_k

        self.linear = torch.nn.Linear(hidden_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x):
        embedding = self.embed(x)
        output, _ = self.rnn(embedding)

        doc_embedding = output.transpose(1, 2)
        pooled_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(doc_embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)

        doc_embedding = torch.cat(pooled_outputs, 1)

        return self.dropout(self.linear(doc_embedding))


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if nonlinearity == 'tanh':
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if s is None:
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

from torch.autograd import Variable

class HAN(nn.Module):
    def __init__(self, vocab_dict, glove_file, emb_dim, dropout_p=0.5, word_embed_dim=300):
        super(HAN, self).__init__()
        self.num_tokens = len(vocab_dict)
        self.embed_size = word_embed_dim
        self.word_gru_hidden = 50
        self.lookup = nn.Embedding(self.num_tokens, self.embed_size)
        if glove_file:
            embeddings = load_glove_embeddings(
                glove_file, vocab_dict, embedding_dim=word_embed_dim
            )
            self.lookup.weight = nn.Parameter(embeddings)
        self.word_gru = nn.GRU(self.embed_size, self.word_gru_hidden, bidirectional=True)
        self.weight_W_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 2 * self.word_gru_hidden))
        self.bias_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))
        nn.init.uniform_(self.weight_W_word, -0.1, 0.1)
        nn.init.uniform_(self.bias_word, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj_word, -0.1, 0.1)
        # sentence level
        self.sent_gru_hidden = 50
        self.word_gru_hidden = 50
        self.sent_gru = nn.GRU(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
        self.weight_W_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 2 * self.sent_gru_hidden))
        self.bias_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
        self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
        C = emb_dim
        self.fc1 = nn.Linear(2 * self.sent_gru_hidden, C)
        nn.init.uniform_(self.bias_sent, -0.1, 0.1)
        nn.init.uniform_(self.weight_W_sent, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj_sent, -0.1, 0.1)

    def forward(self, mini_batch):
        mini_batch = mini_batch.unsqueeze(0)
        max_sents, batch_size, max_tokens = mini_batch.size()
        word_attn_vectors = None
        state_word = self.init_hidden(mini_batch.size()[1])
        for i in range(max_sents):
            embed = mini_batch[i, :, :].transpose(0, 1)
            embedded = self.lookup(embed)
            output_word, state_word = self.word_gru(embedded, state_word)
            word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
            # logger.debug(word_squish.size()) torch.Size([20, 2, 200])
            word_attn = batch_matmul(word_squish, self.weight_proj_word)
            # logger.debug(word_attn.size()) torch.Size([20, 2])
            word_attn_norm = F.softmax(word_attn.transpose(1, 0), dim=-1)
            word_attn_vector = attention_mul(output_word, word_attn_norm.transpose(1, 0))
            if word_attn_vectors is None:
                word_attn_vectors = word_attn_vector
            else:
                word_attn_vectors = torch.cat((word_attn_vectors, word_attn_vector), 0)
        # logger.debug(word_attn_vectors.size()) torch.Size([1, 2, 200])
        state_sent = self.init_hidden(mini_batch.size()[1])
        output_sent, state_sent = self.sent_gru(word_attn_vectors, state_sent)
        # logger.debug(output_sent.size()) torch.Size([8, 2, 200])
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        # logger.debug(sent_squish.size()) torch.Size([8, 2, 200])
        if len(sent_squish.size()) == 2:
            sent_squish = sent_squish.unsqueeze(0)
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        if len(sent_attn.size()) == 1:
            sent_attn = sent_attn.unsqueeze(0)
        # logger.debug(sent_attn.size())  torch.Size([8, 2])
        sent_attn_norm = F.softmax(sent_attn.transpose(1, 0), dim=-1)
        # logger.debug(sent_attn_norm.size()) torch.Size([2, 8])
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1, 0))
        # logger.debug(sent_attn_vectors.size()) torch.Size([1, 2, 200])
        x = sent_attn_vectors.squeeze(0)
        x = self.fc1(x)
        return x

    def init_hidden(self, batch_size, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = self.sent_gru_hidden
        return Variable(torch.zeros(2, batch_size, hidden_dim)).cuda()


class OHCNN_fast(nn.Module):
    def __init__(self, vocab_dict, glove_file, emb_dim, dropout_p=0.5, word_embed_dim=300):
        super(OHCNN_fast, self).__init__()
        D = len(vocab_dict)        
        Co = 1000 #word_embed_dim
        self.Co = Co
        self.n_pool = 10
        self.embed = nn.Embedding(D, Co)
        if glove_file:
            embeddings = load_glove_embeddings(
                glove_file, vocab_dict, embedding_dim=1000
            )
            self.embed.weight = nn.Parameter(embeddings)
        self.bias = nn.Parameter(torch.Tensor(1, Co, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(self.Co * self.n_pool, emb_dim)
        self.unk_idx = vocab_dict['UNK']
        # init as in cnn
        stdv = 1. / np.sqrt(D)
        self.embed.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        sent_len = x.shape[1]
        x = x.view(x.shape[0], -1)
        x_embed = self.embed(x)  # (N, W * D, Co)
        # deal with unk in the region
        x = (x != self.unk_idx).float().unsqueeze(-1) * x_embed
        x = x.view(x.shape[0], sent_len, -1, self.Co)  # (N, W, D, Co)
        x = F.relu(x.sum(2).permute(0, 2, 1) + self.bias)  # (N, Co, W)
        x = F.avg_pool1d(x, int(x.shape[2] / self.n_pool)).view(-1, self.n_pool * self.Co)  # (N, n_pool * Co)
        x = self.dropout(x)
        # response norm
        t = x.clone()
        t = torch.sqrt(1+torch.sum(t*t, dim=1))
        x /= t.view(-1, 1)
        x = self.fc(x)  # (N, C)
        return x


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
    def __init__(self, n_labels, emb_dim=104, dropout_p=0.4):
        super(LabelEmbedModel, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.e = nn.Embedding(
                    n_labels, emb_dim,
                    # max_norm=1.0,
                    # sparse=True,
                    scale_grad_by_freq=False
                )
        self.init_weights()

    def init_weights(self, scale=1e-4):
        torch.nn.init.eye_(self.e.weight)

    def forward(self, x):
        """
        x is the list of labels eg [1,3,8]
        """
        return self.dropout(self.e(x))
