import torch
import matplotlib.pyplot as plt
from matplotlib import collections as mc

t = torch.load('Poin/29')
h = t['label_embs']

h = h/(1+torch.sqrt(1+h.norm(dim=-1, keepdim=True)**2))
h = h.cpu().detach().numpy()

edges = [[0,16], [1,16], [4,16], [5,16], [2,17], [3,17], [6,17], [7,17], [8,18], [9,18], [12,18], [13,18], [10,19], [11,19], [14,19], [15,19], [16, 20], [17, 20], [18, 20], [19,20]]
lines = [[h[i], h[j]] for i,j in edges if i<34 and j < 34]

lc = mc.LineCollection(lines, color='gray', linewidths=0.5)

fig, ax = plt.subplots()
ax.scatter(*h.T)
ax.add_collection(lc)

for i in range(21):
    ax.annotate(str(i), h[i])

plt.savefig('poin')