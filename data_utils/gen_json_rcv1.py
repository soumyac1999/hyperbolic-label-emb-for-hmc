import json
from tqdm import tqdm
from readData_rcv1 import read_rcv1

X_train, X_test, train_ids, test_ids, id2doc, nodes = read_rcv1()

lab_f = open('../rcv1/rcv1_labels.txt', 'w')
tax_f = open('../rcv1/rcv1.taxonomy', 'w')
for node in nodes.keys():
    lab_f.write(node+'\n')
    if len(nodes[node]['children'])>0:
        tax_f.write(node+'\t'+'\t'.join(nodes[node]['children'])+'\n')
lab_f.close()
tax_f.close()
num_labels, max_labels = 0,0
with open('../rcv1/rcv1_train.json', 'w') as f:
    for i, idx in enumerate(tqdm(train_ids)):
        x = {}
        x["label"] = id2doc[idx]['categories']
        x["text"] = X_train[i]
        print(x["label"],x["text"])
        f.write(json.dumps(x)+"\n")

with open('../rcv1/rcv1_test.json', 'w') as f:
    for i, idx in enumerate(tqdm(test_ids)):
        x = {}
        x["label"] = id2doc[idx]['categories']
        x["text"] = X_test[i]
        f.write(json.dumps(x)+"\n")
