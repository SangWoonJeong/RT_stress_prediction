
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import pandas as pd
import numpy as np

from sklearn.metrics import multilabel_confusion_matrix, mean_absolute_error, mean_absolute_percentage_error, \
    accuracy_score
from sklearn import metrics

folder_root = "/home/smc-gpu/PycharmProjects/"


#
for iter99 in range(3):

    all_result_list = []
    all_result_data_list = []

    for iter00 in range(1):
        print(iter00 + 1)

        train_file = f'fold_{iter00 + 1:02d}_train.xlsx'
        test_file = f'fold_{iter00 + 1:02d}_test.xlsx'

        all_train_dataset = pd.read_excel(folder_root + train_file)
        all_test_dataset = pd.read_excel(folder_root + test_file)

        all_train_dataset = all_train_dataset.values
        all_test_dataset = all_test_dataset.values
        #  input 1:age, 2:sex, 3:day, 4-10: feature
        # output 11-17: feature, 18:stress

        if iter99==0: # Age + day + features (type 6)
            train_x = np.concatenate([all_train_dataset[:, 4:11], all_train_dataset[:, 1:2], all_train_dataset[:, 3:4]],
                                     axis=1)  # Age + day + f
            test_x = np.concatenate([all_test_dataset[:, 4:11], all_test_dataset[:, 1:2], all_test_dataset[:, 3:4]],
                                     axis=1)  # Age + day + f

        elif iter99==1: # Sex + day + features (type 7)
            train_x = np.concatenate([all_train_dataset[:, 4:11], all_train_dataset[:, 2:4]], axis=1)  # Sex + day + f
            test_x = np.concatenate([all_test_dataset[:, 4:11], all_test_dataset[:, 2:4]], axis=1)  # Sex + day + f

        elif iter99==2: # Age + sex + day + features (type 8)
            train_x = np.concatenate([all_train_dataset[:, 4:11], all_train_dataset[:, 1:4]],
                                     axis=1)  # Age + sex + day + f
            test_x = np.concatenate([all_test_dataset[:, 4:11], all_test_dataset[:, 1:4]],
                                     axis=1)  # Age + sex + day + f


        train_y = all_train_dataset[:, 11:18]
        test_y = all_test_dataset[:, 11:18]

        atrain_x = train_x
        atest_x = test_x
        atrain_y = train_y
        atest_y = test_y

        for iter01 in range(10):
            # iter01 = 0
            print(iter99+1 ,iter00 + 1, iter01 + 1)
            # Create the transformer model
            #model = DecisionTreeClassifier() #DT
            #model = RandomForestClassifier() #RF
            #model.fit(atrain_x, atrain_y)
            #pred_data = model.predict(atest_x)


            model = SVC()  #SVC
            multi_label_model = OneVsRestClassifier(model)
            multi_label_model.fit(atrain_x, atrain_y)
            pred_data = multi_label_model.predict(atest_x)

            y_pred_f = pred_data
            eval_AS = accuracy_score(y_pred_f, atest_y)

            correct_prediction = pred_data == atest_y
            accuracy = correct_prediction.mean()

            qqqq2 = multilabel_confusion_matrix(y_pred_f, atest_y)
            eval_confu_precision_ma = metrics.precision_score(y_pred_f, atest_y, average='macro')
            eval_confu_precision_mi = metrics.precision_score(y_pred_f, atest_y, average='micro')
            eval_confu_acc = metrics.accuracy_score(y_pred_f, atest_y)
            eval_confu_recall_ma = metrics.recall_score(y_pred_f, atest_y, average='macro', zero_division=0)
            eval_confu_recall_mi = metrics.recall_score(y_pred_f, atest_y, average='micro', zero_division=0)
            eval_confu_f1_ma = metrics.f1_score(y_pred_f, atest_y, average='macro')
            eval_confu_f1_mi = metrics.f1_score(y_pred_f, atest_y, average='micro')

            # MAE
            pre_array = y_pred_f
            true_array = atest_y

            pred_rate = pre_array.sum(axis=1)
            true_rate = true_array.sum(axis=1)

            ddd = pred_rate * 14.286
            ddd2 = true_rate * 14.286

            eval_mae = mean_absolute_error(ddd, ddd2)
            accuracy_f = accuracy

            evaluation_sum = [eval_confu_acc, eval_mae, accuracy_f, eval_confu_precision_ma,
                              eval_confu_precision_mi, eval_confu_recall_ma, eval_confu_recall_mi, eval_confu_f1_ma,
                              eval_confu_f1_mi]

            all_result_list.append(evaluation_sum)
            all_result_data_list.append(pred_data)

    fff_results = pd.DataFrame.from_records(all_result_list)
    all_r_file = f'SVC_2_feature_eval_{iter99+1:02d}.xlsx'
    fff_results.to_excel(all_r_file, index=False)

    db_fff_results = pd.DataFrame.from_records(all_result_data_list)
    db_all_r_file = f'SVC_2_feature_DB_{iter99+1:02d}.xlsx'
    db_fff_results.to_excel(db_all_r_file, index=False)





























