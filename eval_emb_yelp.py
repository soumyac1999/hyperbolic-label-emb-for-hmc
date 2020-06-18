import torch
import numpy as np
# from sklearn.metrics import ndcg_score, dcg_score
from avg_prec import dcg_score,ndcg_score
from poincare_utils import PoincareDistance
from tqdm import tqdm

try:
	hops = np.load('yelp/hops.npy')
	n_labels = hops.shape[0]
except:
	label_file = 'yelp/rcv_labels.txt'
	hier_file = 'yelp/rcv.taxonomy'

	labels = [l.strip() for l in open(label_file)]
	stoi = {l: i for i, l in enumerate(labels)}
	n_labels = len(labels)


	hops = np.zeros((n_labels, n_labels), dtype=np.float32) + np.inf
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
	
	hops[hops==np.inf]=-np.inf
	m = 2*np.max(hops)
	hops[hops==-np.inf]= m

	np.save('yelp/hops', hops)

num_ndcg_k = 6
#rcv-17,8,14,29
#nyt- 2,2 4,2
#nyt-macro - 21, 7, 26, 22
#yelp - 49, 47, 48, 42
#yelp macro - 49, 46, 46,4
emb = torch.load(f'yelp_exp1/flat_textcnn_check/10')['label_embs']
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(1,n_labels):
		for j in range(n_labels):
			pred[i][j] = (emb[i]-emb[j]).norm()

flat = []
from avg_prec import dcg_score
for j in range(1,num_ndcg_k):
	x=0
	for i in range(n_labels):
		x += ndcg_score(hops[:,i], pred[:,i].numpy(),k=j, gains="hops")
	flat.append(x/n_labels)





emb = torch.load(f'yelp_exp1/poin_0.1_textcnn/46')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

jt = []
for j in range(1,num_ndcg_k):
	x=0
	for i in range(n_labels):
		x += ndcg_score(hops[:,i], pred[:,i].numpy(),k=j, gains="hops")
	jt.append(x/n_labels)


emb = torch.load(f'yelp_exp1/poin_0.01_textcnn/46')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

jt01 = []
for j in range(1,num_ndcg_k):
	x=0
	for i in range(n_labels):
		x += ndcg_score(hops[:,i], pred[:,i].numpy(),k=j, gains="hops")
	jt01.append(x/n_labels)



emb = torch.load(f'yelp_exp1/only_label_emb/4')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

lab = []
for j in range(1,num_ndcg_k):
	x=0
	for i in range(n_labels):
		x += ndcg_score(hops[:,i], pred[:,i].numpy(),k=j, gains="hops")
	lab.append(x/n_labels)

print(flat,jt,jt01,lab)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(np.arange(1,num_ndcg_k), jt, label="HIDDEN_jnt 0.1")
plt.plot(np.arange(1,num_ndcg_k), jt01, label="HIDDEN_jnt 0.01")
plt.plot(np.arange(1,num_ndcg_k), lab, label="HIDDEN_cas")
plt.xlabel("k")
plt.ylabel("NDCG")
plt.xticks([1,2,3,4,5])
plt.legend()
plt.savefig('yelp_emb.png')
