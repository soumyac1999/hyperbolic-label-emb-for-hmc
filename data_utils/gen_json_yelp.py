import json
import random
from tqdm import tqdm

random.seed(42)

labels = set()

for l in open('yelp/Taxonomy_100'):
    t = eval(l)
    labels.update([t['title'].replace(' ', '_')])

labels.remove('root')

business = {}
rewiews = {}
no_cats = 0

for l in open('yelp/yelp_academic_dataset_business.json'):
    t = json.loads(l)
    try:
        categories = [c.strip().replace(' ', '_') for c in t['categories'].split(',')]
        categories = set(categories)
        business[t['business_id']] = list(categories.intersection(labels))
        rewiews[t['business_id']] = []
    except:
        no_cats += 1

print("%i businesses don't have categories. Using %i" % (no_cats, len(business)))

with open('yelp/yelp_labels.txt', 'w') as f:
    for i in labels:
        f.write(i+'\n')

review_no_cats = 0

for l in open('yelp/yelp_academic_dataset_review.json'):
    t = json.loads(l)
    try:
        rewiews[t['business_id']].append(t['text'])
    except:
        review_no_cats += 1

X = []

for business_id in business.keys():
    if len(rewiews[business_id])>=5:
        X.append(('\t'.join(rewiews[business_id][:10]), business[business_id]))

print("%i reviews not used since business categories doen't exist. Using %i" % (review_no_cats, len(X)))

random.shuffle(X)

test = int(0.3*len(X))

print("%i training examples and %i test examples" % (len(X)-test, test))

with open('yelp/yelp_test.json', 'w') as f:
    for i in tqdm(range(0, test)):
        x = {}
        x["label"] = X[i][1]
        x["text"] = X[i][0]
        f.write(json.dumps(x)+"\n")

with open('yelp/yelp_train.json', 'w') as f:
    for i in tqdm(range(test, len(X))):
        x = {}
        x["label"] = X[i][1]
        x["text"] = X[i][0]
        f.write(json.dumps(x)+"\n")
