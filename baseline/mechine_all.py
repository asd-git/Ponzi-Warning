import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix, classification_report
from xgboost import XGBClassifier

from utils.utils import load_data_machine, load_data_machine_new, load_data_machine_all_feature

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(1)  # reproducible
def diff_index(test_y, pred_y):
    diff_index = [index for index,(item1,item2) in enumerate(zip(test_y, pred_y)) if item1 != item2]
    return diff_index

# 加载数据集
edge_num_ls = list(range(10,110,10))
kfolds = 5
seed = 111
feature_type = 'all_trans_code_arctan'
rnd_states = [
    seed - 111
]

for rnd_state in rnd_states:
    np.random.seed(rnd_state)
    for edge_num in edge_num_ls:

        # train_test_kfold_dataset = load_data_machine(edge_num,edge_num_ls, rnd_state, kfolds)
        train_test_kfold_dataset = load_data_machine_all_feature(edge_num,edge_num_ls, rnd_state, kfolds)

        pre_folds_dict = {
            'XGB': [],
            'ada': [],
            'rfc': [],
            'gbc': [],
        }
        recall_folds_dict = {
            'XGB': [],
            'ada': [],
            'rfc': [],
            'gbc': [],
        }
        f1_folds_dict = {
            'XGB': [],
            'ada': [],
            'rfc': [],
            'gbc': [],
        }

        we_pre_folds_dict = {
            'XGB': [],
            'ada': [],
            'rfc': [],
            'gbc': [],
            'mlp': []
        }
        we_recall_folds_dict = {
            'XGB': [],
            'ada': [],
            'rfc': [],
            'gbc': [],
            'mlp': []
        }
        we_f1_folds_dict = {
            'XGB': [],
            'ada': [],
            'rfc': [],
            'gbc': [],
            'mlp': []
        }
        for kf in range(kfolds):
            data = train_test_kfold_dataset[kf]
            all_train_x = data[0][0]

            train_y = data[0][1]

            all_test_x = data[1][0]

            test_y = data[1][1]


            tuned_parameters = {
                "XGB": [
                    {'n_estimators': range(80, 200, 4),
                     'max_depth': range(2, 15, 1),
                     'learning_rate': np.linspace(0.01, 2, 20),
                     'subsample': np.linspace(0.7, 0.9, 20),
                     'colsample_bytree': np.linspace(0.5, 0.98, 10),
                     'min_child_weight': range(1, 9, 1)}
                ],
                'ada': [{'n_estimators': range(10, 500, 10),
                         'learning_rate': np.linspace(0.01, 2, 20)
                         }],
                'rfc': [{"n_estimators": range(10, 500, 10),
                         "criterion": ["gini", "entropy"],
                         'max_depth': range(1, 20, 1),
                         'min_samples_leaf': range(1, 10, 1),
                         }],
                'gbc': [{"n_estimators": range(10, 500, 10),
                         'max_depth': range(1, 20, 1),
                         'min_samples_leaf': range(1, 10, 1),
                         # 'min_samples_split': [0,1,2,3],
                         }],

            }
            clfs = {'XGB': XGBClassifier(),
                    'ada': AdaBoostClassifier(),
                    'rfc': RandomForestClassifier(),
                    # 'gbc': GradientBoostingClassifier(),
                    }

            f1_scores = dict()
            pre_scores = dict()
            recall_scores = dict()
            we_f1_scores = dict()
            we_pre_scores = dict()
            we_recall_scores = dict()
            for clf_name in clfs:
                print(clf_name)
                gsc = RandomizedSearchCV(clfs[clf_name], tuned_parameters[clf_name])
                grid_result = gsc.fit(all_train_x, train_y)
                print("Best parameters : ", grid_result.best_params_)
                # Predict..
                print("test")
                y_pred = grid_result.predict(all_test_x)

                # clf = clfs[clf_name]
                # clf.fit(trans_train_x, train_y)
                # y_pred = clf.predict(trans_test_x)

                # Evaluate the model
                print(classification_report(test_y, y_pred))
                pre_scores[clf_name] = precision_score(test_y, y_pred, average='binary')
                pre_folds_dict[clf_name].append(pre_scores[clf_name])
                recall_scores[clf_name] = recall_score(test_y, y_pred, average='binary')
                recall_folds_dict[clf_name].append(recall_scores[clf_name])
                f1_scores[clf_name] = f1_score(test_y, y_pred, average='binary')
                f1_folds_dict[clf_name].append(f1_scores[clf_name])

                we_pre_scores[clf_name] = precision_score(test_y, y_pred, average='weighted')
                we_pre_folds_dict[clf_name].append(we_pre_scores[clf_name])
                we_recall_scores[clf_name] = recall_score(test_y, y_pred, average='weighted')
                we_recall_folds_dict[clf_name].append(we_recall_scores[clf_name])
                we_f1_scores[clf_name] = f1_score(test_y, y_pred, average='weighted')
                we_f1_folds_dict[clf_name].append(we_f1_scores[clf_name])


                diff_index_ls = diff_index(test_y, y_pred)
                print(diff_index_ls)


            print("pre", pre_scores)
            print("recall", recall_scores)
            print("f1_score", f1_scores)


        for clf_name in clfs:
            mean_f1 = np.mean(f1_folds_dict[clf_name])
            std_f1 = np.std(f1_folds_dict[clf_name])
            mean_prec = np.mean(pre_folds_dict[clf_name])
            mean_rec = np.mean(recall_folds_dict[clf_name])
            print(clf_name)

            print("Mean f1:{}".format(mean_f1))
            print("Std f1:{}".format(std_f1))
            print("Mean precision:{}".format(mean_prec))
            print("Mean recall:{}".format(mean_rec))

            we_mean_f1 = np.mean(we_f1_folds_dict[clf_name])
            we_std_f1 = np.std(we_f1_folds_dict[clf_name])
            we_mean_prec = np.mean(we_pre_folds_dict[clf_name])
            we_mean_rec = np.mean(we_recall_folds_dict[clf_name])
            print(clf_name)

            print("Mean f1:{}".format(we_mean_f1))
            print("Std f1:{}".format(we_std_f1))
            print("Mean precision:{}".format(we_mean_prec))
            print("Mean recall:{}".format(we_mean_rec))

            savepath = "./result/machine/{}/".format(feature_type)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            with open(savepath + "/result_binary.txt", 'a')as w:
                w.writelines(
                    "rnd_state:{}  kfolds:{}  clf:{}  edge_num:{}  Mean pre--Mean recall--Mean f1--Std f1: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                        rnd_state, kfolds, clf_name, edge_num, mean_prec, mean_rec, mean_f1,
                        std_f1))

            with open(savepath + "/result_weighted.txt", 'a')as w:
                w.writelines(
                    "rnd_state:{}  kfolds:{}  clf:{}  edge_num:{}  Mean pre--Mean recall--Mean f1--Std f1: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                        rnd_state, kfolds, clf_name, edge_num, we_mean_prec, we_mean_rec, we_mean_f1,
                        we_std_f1))
