# %%

import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, mean_absolute_error, mean_absolute_percentage_error, \
    accuracy_score
from sklearn import metrics


torch.cuda.is_available()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(0)
torch.cuda.current_device()



class MultiLabelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelLSTM, self).__init__()
        self.hidden_size = hidden_size

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)

        # 출력층
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_hidden_state = lstm_out[:,-1,:]
        logits = self.linear(last_hidden_state)
        return torch.sigmoid(logits)


folder_root = ""

for iter99 in range(3):
    # iter99 = 0 1 2

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

        if iter99==0:
            train_x = np.concatenate([all_train_dataset[:, 4:11], all_train_dataset[:, 1:2], all_train_dataset[:, 3:4]],
                                     axis=1)  # Age + day + f
            test_x = np.concatenate([all_test_dataset[:, 4:11], all_test_dataset[:, 1:2], all_test_dataset[:, 3:4]],
                                     axis=1)  # Age + day + f

        elif iter99==1:
            train_x = np.concatenate([all_train_dataset[:, 4:11], all_train_dataset[:, 2:4]], axis=1)  # Sex + day + f
            test_x = np.concatenate([all_test_dataset[:, 4:11], all_test_dataset[:, 2:4]], axis=1)  # Sex + day + f

        elif iter99==2:
            train_x = np.concatenate([all_train_dataset[:, 4:11], all_train_dataset[:, 1:4]],
                                     axis=1)  # Age + sex + day + f
            test_x = np.concatenate([all_test_dataset[:, 4:11], all_test_dataset[:, 1:4]],
                                     axis=1)  # Age + sex + day + f


        train_y = all_train_dataset[:, 11:18]
        test_y = all_test_dataset[:, 11:18]

        atrain_x = torch.FloatTensor(np.expand_dims(train_x, -1))
        atest_x = torch.FloatTensor(np.expand_dims(test_x, -1))
        atrain_y = torch.LongTensor(train_y.squeeze())
        atest_y = torch.LongTensor(test_y.squeeze())


        input_size = 1
        num_classes = 7
        hidden_size = 64  # 30

        for iter01 in range(10):
            # iter01 = 0
            print(iter99+1 ,iter00 + 1, iter01 + 1)
            # Create the transformer model
            model = MultiLabelLSTM(input_size, hidden_size, num_classes).to(device)
            lr = 0.0005

            # Define loss function using class weights
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            cost_list = []
            test_list = []
            vali_list = []

            loss_hist = {'train_cost': [],
                         'test_cost': []}

            num_epochs = 20000

            for epoch in range(num_epochs):
                model.train()
                temp_input = atrain_x[:, :, 0]
                main_input = temp_input.unsqueeze(2)
                output = model(main_input.to(device))
                cost = criterion(output, atrain_y.to(device).float())

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    temp_test_input = atest_x[:, :, 0]
                    main_test_input = temp_test_input.unsqueeze(2)
                    outputs = model(main_test_input.to(device))
                    cost_test = criterion(outputs, atest_y.to(device).float())
                    test_list.append(outputs)

                loss_hist['train_cost'].append(cost.item())
                loss_hist['test_cost'].append(cost_test.item())
                if epoch % 200 == 0:
                    print(f"Epoch : {epoch + 1:4d}, train_Cost : {cost:.3f}, test_Cost : {cost_test:.3f}")

            test_min_loss_n = loss_hist['test_cost'].index((np.min(loss_hist['test_cost'])))
            pred_data = test_list[test_min_loss_n]

            y_pred_f = torch.round(pred_data)
            y_true_f = torch.round(output)

            eval_AS = accuracy_score(y_pred_f.cpu(), atest_y)

            correct_prediction = torch.round(pred_data) == atest_y.to(device)
            accuracy = correct_prediction.float().mean()

            qqqq2 = multilabel_confusion_matrix(y_pred_f.cpu(), atest_y)
            eval_confu_precision_ma = metrics.precision_score(y_pred_f.cpu(), atest_y, average='macro')
            eval_confu_precision_mi = metrics.precision_score(y_pred_f.cpu(), atest_y, average='micro')
            eval_confu_acc = metrics.accuracy_score(y_pred_f.cpu(), atest_y)
            eval_confu_recall_ma = metrics.recall_score(y_pred_f.cpu(), atest_y, average='macro', zero_division=0)
            eval_confu_recall_mi = metrics.recall_score(y_pred_f.cpu(), atest_y, average='micro', zero_division=0)
            eval_confu_f1_ma = metrics.f1_score(y_pred_f.cpu(), atest_y, average='macro')
            eval_confu_f1_mi = metrics.f1_score(y_pred_f.cpu(), atest_y, average='micro')

            # MAE
            pre_array = np.array(y_pred_f.cpu())
            true_array = np.array(atest_y.cpu())

            pred_rate = pre_array.sum(axis=1)
            true_rate = true_array.sum(axis=1)

            ddd = pred_rate * 14.286
            ddd2 = true_rate * 14.286

            eval_mae = mean_absolute_error(ddd, ddd2)
            accuracy_f = np.array(accuracy.cpu())

            evaluation_sum = [eval_confu_acc, eval_mae, accuracy_f.min(), eval_confu_precision_ma,
                              eval_confu_precision_mi, eval_confu_recall_ma, eval_confu_recall_mi, eval_confu_f1_ma,
                              eval_confu_f1_mi]

            all_result_list.append(evaluation_sum)
            all_result_data_list.append(np.array(pred_data.cpu()))

    fff_results = pd.DataFrame.from_records(all_result_list)
    all_r_file = f'LSTM_2_feature_eval_{iter99+1:02d}.xlsx'
    fff_results.to_excel(all_r_file, index=False)

    db_fff_results = pd.DataFrame.from_records(all_result_data_list)
    db_all_r_file = f'LSTM_2_feature_DB_{iter99+1:02d}.xlsx'
    db_fff_results.to_excel(db_all_r_file, index=False)


























