import glob
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import random

MAX_DEPTH = 3

MAX_DEPTH += 1

random.seed(42)

class Hier(object):
	def __init__(self, label):
		super(Hier, self).__init__()
		self.label = label
		self.children = {}

	def __repr__(self):
		return self.label + '\t' + '\t'.join(k for k in self.children)

	def __getitem__(self, key):
		ret = None
		try:
			ret = self.children[key]
		except KeyError:
			self.children[key] = Hier(key)
			ret = self.children[key]
		return ret

	def write_hier(self, fd):
		if len(self.children) > 0:
			fd.write(str(self))
			fd.write('\n')
			for _, v in self.children.items():
				v.write_hier(fd)

total = 0
skipped = 0
train = 0
test = 0
all_labels = set()

train_file = open('nyt/nyt_train.json', 'w')
test_file = open('nyt/nyt_test.json', 'w')
error_file = open('nyt_err.txt', 'w')

hier_root = Hier('Root')
for file in tqdm(glob.glob('../nyt_corpus/data/*/*/*/*.xml')):
	tree = ET.parse(file)
	root = tree.getroot()
	meta = root.find('head').find('docdata')\
			.find('identified-content').findall('classifier')
	labels = [t.text.replace(' ', '_').split('/')[:MAX_DEPTH] for t in meta 
				if t.attrib['type']=='taxonomic_classifier']
	blocks = root.find('body').find('body.content').findall('block')
	lead_para = [t for t in blocks if t.attrib['class']=='lead_paragraph']
	if len(lead_para) != 1:
		skipped += 1
		error_file.write('Skipped %s: No lead-paragraph found\n'%(file))
		continue
	text = ' '.join([p.text for p in lead_para[0]])
	label = list(set(l for t in labels for l in t))
	if len(label) == 0:
		skipped += 1
		error_file.write('Skipped %s: No labels found\n'%(file))
		continue
	x = {}
	x["label"] = label
	x["text"] = text
	all_labels.update(label)
	total += 1
	if random.random() > 0.3:
		train_file.write(json.dumps(x)+"\n")
		train += 1
	else:
		test_file.write(json.dumps(x)+"\n")
		test += 1
	for label in labels:
		t_root = hier_root
		for i in label:
			t_root = t_root[i]

print('Total Docs: %i, Skipped: %i, Train: %i, Test: %i' \
	% (total, skipped, train, test))

hier_root.write_hier(open('nyt/nyt.taxonomy', 'w'))

with open('nyt/nyt_labels.txt', 'w') as f:
    for i in all_labels:
        f.write(i+'\n')
