import torch
import numpy as np
from sklearn.metrics import ndcg_score
from poincare_utils import PoincareDistance
from tqdm import tqdm

try:
	hops = np.load('rcv1/hops.npy')
	n_labels = hops.shape[0]
except:
	label_file = 'rcv1/rcv1_labels.txt'
	hier_file = 'rcv1/rcv1.taxonomy'

	labels = [l.strip() for l in open(label_file)]
	stoi = {l: i for i, l in enumerate(labels)}
	n_labels = len(labels)

	hops = torch.zeros((n_labels, n_labels)) + np.inf
	for l in open(hier_file):
	    ls = l.split('\t')
	    h = ls[0]
	    t = ls[1:]
	    h = stoi[h.strip()]
	    t = [stoi[v.strip()] for v in t]
	    hops[h, t] = 1
	    hops[t, h] = 1

	for i in range(n_labels):
		hops[i, i] = 0

	for k in tqdm(range(n_labels)):
		for i in range(n_labels):
			for j in range(n_labels):
				hops[i, j] = min(hops[i, j], hops[i, k] + hops[k, j])

	hops = hops.numpy()
	np.save('rcv1/hops', hops)

# import pickle
# hops = pickle.load(open('rcv1/train.pkl', 'rb')).text_dataset.similarity.numpy()

# Flat macro
emb = torch.load(f'rcv1_exp/flat_textcnn/26')['label_embs']
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = (emb[i]-emb[j]).norm()

print(ndcg_score(hops, pred.numpy(), k=3))

# Flat micro
emb = torch.load(f'rcv1_exp/flat_textcnn/16')['label_embs']
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = (emb[i]-emb[j]).norm()

print(ndcg_score(hops, pred.numpy(), k=3))

# Poin macro
emb = torch.load(f'rcv1_exp/poin_0.1_textcnn/25')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

print(ndcg_score(hops, pred.numpy(), k=3))

# Poin micro
emb = torch.load(f'rcv1_exp/poin_0.1_textcnn/7')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

print(ndcg_score(hops, pred.numpy(), k=3))

# Only label
emb = torch.load(f'rcv1_exp/only_label_emb/29')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

print(ndcg_score(hops, pred.numpy(), k=3))


# for exp in ['flat_textcnn', 'poin_0.1_textcnn', 'only_label_emb']:
# 	ndcg = []
# 	for i in range(30):
# 		emb = torch.load(f'rcv1_exp/{exp}/{i}')['label_embs']
# 		if 'flat' not in exp:
# 			emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
# 		emb = emb.detach().cpu()

# 		with torch.no_grad():
# 			pred = torch.zeros((n_labels, n_labels))
# 			for i in range(n_labels):
# 				for j in range(n_labels):
# 					if 'flat' not in exp:
# 						pred[i][j] = PoincareDistance.apply(emb[i], emb[j])
# 					else:
# 						pred[i][j] = (emb[i]-emb[j]).norm()

# 		ndcg.append(ndcg_score(hops, -pred.numpy(), k=3))
# 	print(exp, ndcg)
