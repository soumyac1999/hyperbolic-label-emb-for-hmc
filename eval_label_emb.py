import torch
import numpy as np
from sklearn.metrics import ndcg_score, dcg_score
from poincare_utils import PoincareDistance
from tqdm import tqdm

try:
	hops = np.load('rcv1/hops.npy')
	n_labels = hops.shape[0]
except:
	label_file = 'rcv1/rcv_labels.txt'
	hier_file = 'rcv1/rcv.taxonomy'

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
	# print(hops)
	np.save('rcv1/hops', hops)

# import pickle
# hops = pickle.load(open('rcv1/train.pkl', 'rb')).text_dataset.similarity.numpy()

# hops = hops + 1e-5
# rcv- 27,26,29
# nyt - 22, 8,23
#yelp - 49, 47,5
# # Flat macro
emb = torch.load(f'yelp_exp/flat_textcnn/12')['label_embs']
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = (emb[i]-emb[j]).norm()

# pred = pred + 1e-5
x = 0
from avg_prec import dcg_score
for i in range(n_labels):
	x += dcg_score(hops[:,i], pred[:,i].numpy(),k=5, gains="hops")
print(x/n_labels)

# print('predictiosn are ',1/pred)
# print('hops are ',1/hops)

# print("Flat macro ndcg@5 is ", ndcg_score(1./hops, 1./pred.numpy(), k=5))
# print("Flat macro ndcg@3 is ", ndcg_score(1./hops, 1./pred.numpy(), k=3))
# print("Flat macro ndcg@1 is ", ndcg_score(1./hops, 1./pred.numpy(), k=1))
# average_precision = dict()

# # for i in range(n_labels):
# # 	average_precision[i] = average_precision_score(np.round(pred[:, i]), hops[:, i])
# # average_precision["macro"] = average_precision_score(pred, hops,average="macro")
# # print('average_precision_score' , label_ranking_average_precision_score(hops, pred.numpy()))
# # Flat micro
# emb = torch.load(f'yelp_exp/flat_textcnn/17')['label_embs']
# emb = emb.detach().cpu()

# with torch.no_grad():
# 	pred = torch.zeros((n_labels, n_labels))
# 	for i in range(n_labels):
# 		for j in range(n_labels):
# 			pred[i][j] = (emb[i]-emb[j]).norm()
# # print('predictiosn are ',pred)
# print(" Flat micro ndcg@5 is ", ndcg_score(hops, pred.numpy()))
# print("Flat micro ndcg@3 is ",ndcg_score(hops, pred.numpy()))
# print("Flat micro ndcg@1 is ",ndcg_score(hops, pred.numpy()))

# Poin macro
emb = torch.load(f'yelp_exp/poin_0.1_textcnn/27')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

# pred = pred + 1e-5
x=0
for i in range(n_labels):
	x += dcg_score(hops[:,i], pred[:,i].numpy(),k=5, gains="hops")
print(x/n_labels)

# print("Poin macro ndcg is ", ndcg_score(1./hops, 1./pred.numpy()))
# print("Poin macro ndcg@5 is ", ndcg_score(1./hops, 1./pred.numpy(), k=5))
# print("Poin macro ndcg@3 is ", ndcg_score(1./hops, 1./pred.numpy(), k=3))
# print("Poin macro ndcg@1 is ", ndcg_score(1./hops, 1./pred.numpy(), k=1))

# # Poin micro
# emb = torch.load(f'yelp_exp/poin_0.1_textcnn/8')['label_embs']
# emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
# emb = emb.detach().cpu()

# with torch.no_grad():
# 	pred = torch.zeros((n_labels, n_labels))
# 	for i in range(n_labels):
# 		for j in range(n_labels):
# 			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

# print(" Poin micro ndcg@5 is ", ndcg_score(hops, pred.numpy()))
# print("Poin micro ndcg@3 is ",ndcg_score(hops, pred.numpy()))
# print("Poin micro ndcg@1 is ",ndcg_score(hops, pred.numpy()))

# Only label
emb = torch.load(f'yelp_exp/only_label_emb/5')['label_embs']
emb = emb/(1+torch.sqrt(1+emb.norm(dim=-1, keepdim=True)**2))
emb = emb.detach().cpu()

with torch.no_grad():
	pred = torch.zeros((n_labels, n_labels))
	for i in range(n_labels):
		for j in range(n_labels):
			pred[i][j] = PoincareDistance.apply(emb[i], emb[j])

# pred = pred + 1e-5
x=0
for i in range(n_labels):
	x += dcg_score(hops[:,i], pred[:,i].numpy(),k=5, gains="hops")
print(x/n_labels)

# print("Only label - ndcg is ", ndcg_score(1./hops, 1./pred.numpy()))
# print("Only label - ndcg@5 is ", ndcg_score(1./hops, 1./pred.numpy(), k=5))
# print("Only label - ndcg@3 is ", ndcg_score(1./hops, 1./pred.numpy(), k=3))
# print("Only label - ndcg@1 is ", ndcg_score(1./hops, 1./pred.numpy(), k=1))


# for exp in ['flat_textcnn', 'poin_0.1_textcnn', 'only_label_emb']:
# 	ndcg = []
# 	for i in range(25,29):
# 		emb = torch.load(f'yelp_exp/{exp}/{i}')['label_embs']
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

# 		ndcg.append(dcg_score(hops, -pred.numpy()))
# 	print(exp, ndcg)
