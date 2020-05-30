import json,sys
inst,avg,maxi,tot = 0,0,0,0

data = []
trname = "../" + sys.argv[1] + "/"+sys.argv[1]+"_train.json"
testname = "../" + sys.argv[1] + "/"+sys.argv[1]+"_test.json"
print(trname, testname)
with open(trname) as f:
    for l in f:
        x= len(json.loads(l)['label'])
        tot+=x
        if maxi < x:
            maxi = x
        inst +=1 

with open(testname) as f:
    for l in f:
        x= len(json.loads(l)['label'])
        tot += x
        if maxi < x:
            maxi = x
        inst +=1

print('avg labels are', tot/inst)
print('max labels are ', maxi)
print('total instances are ', inst)
