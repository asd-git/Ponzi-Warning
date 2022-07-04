import numpy as np

from model.dataload import mkdir
import json

m = np.array(list(range(0, 400)))

from sklearn.model_selection import KFold, StratifiedKFold
class mydict(dict):
    def __str__(self):
        return json.dumps(self)
n = np.array([1] * 75 + [0] * 325)
seed = 0
np.random.seed(seed)
skf = StratifiedKFold(n_splits=5, shuffle=True)
edge_num_ls = list(range(10, 110, 10))

test_edge_num_idx = []
kfold_train_test = []

for train_index, test_index in skf.split(m, n):
    kfold_train_test.append([train_index, test_index])

kfold_index_dic = mydict()
for i, item in enumerate(kfold_train_test):
    if i not in kfold_index_dic.keys():
        kfold_index_dic[str(i)] = mydict()
        train_index, test_index = item[0], item[1]
        train_all_idx = [j + (400 * i) for i in range(len(edge_num_ls)) for j in train_index]
        kfold_index_dic[str(i)]["train_idx"] = str(train_all_idx)
        kfold_index_dic[str(i)]["test_idx"] = mydict()
        for edge_num in range(len(edge_num_ls)):
            test_edge_idx = []
            for j in test_index:
                test_edge_idx.append(j + (400 * edge_num))

            kfold_index_dic[str(i)]["test_idx"][str((edge_num + 1) * 10)] = str(test_edge_idx)

# print(kfold_index_dic)

print(kfold_index_dic[str(0)])

# outfile = mkdir('/public/zjj/jj/data1/eth/pup/idx_split/')
# with open(outfile + f'Seed{seed}_idx_split.json', 'w') as f:
#     json.dumps(kfold_index_dic)
#
#
# outfile ='/public/zjj/jj/data1/eth/pup/idx_split/' + f'Seed{seed}_idx_split.json'
# with open(outfile) as f:
#     json_data = json.load(f)
#     print(json_data)