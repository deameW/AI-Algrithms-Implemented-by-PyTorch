import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import preprocessing as prepocessing
from sklearn.metrics import mean_squared_error



if __name__ == '__main__':
    # read data
    train_data = pd.read_csv("./data/train_FD001.txt", sep="\s+", header=None)
    test_data = pd.read_csv("./data/test_FD001.txt", sep="\s+", header=None)
    true_rul = pd.read_csv('./data/RUL_FD001.txt', sep = '\s+', header = None)

    """
    parameters for training data
    """
    window_length = 1
    shift = 1
    early_rul = 125

    processed_train_data = []
    processed_train_targets = []

    """
      parameters for test data
    """
    # How many test windows to take for each engine. If set to 1 (this is the default), only last window of
    # test data for each engine are taken. If set to a different number, that many windows from last are taken.
    # Final output is the average of output of all windows.
    num_test_windows = 5 # 即5组测试数据
    processed_test_data = []
    num_test_windows_list = []

    # number of engines
    num_machines = np.min([len(train_data[0].unique()), len(test_data[0].unique())])

    for i in np.arange(1, num_machines + 1):
        # drop the useless columns
        columns_to_be_dropped = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]
        # train_data[train_data[0] == i] : 获取第一个engine 的数据
        temp_train_data = train_data[train_data[0] == i].drop(columns=columns_to_be_dropped).values
        temp_test_data = test_data[test_data[0] == i].drop(columns=columns_to_be_dropped).values

        # Get rul/target value
        temp_train_targets = prepocessing.process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
        # 画图
        # plt.figure(figsize=(12, 6))
        # plt.plot(temp_train_targets)
        # plt.show()

        data_for_a_machine, targets_for_a_machine = prepocessing.process_input_data_with_targets(temp_train_data, temp_train_targets,
                                                                                    window_length=window_length,
                                                                                    shift=shift)

        # Prepare test data
        test_data_for_an_engine, num_windows = prepocessing.process_test_data(temp_test_data, window_length=window_length,
                                                                 shift=shift,
                                                                 num_test_windows=num_test_windows)

        # record train data & test data each round
        processed_train_data.append(data_for_a_machine)
        processed_train_targets.append(targets_for_a_machine)

        processed_test_data.append(test_data_for_an_engine)
        num_test_windows_list.append(num_windows)


    processed_train_data = np.concatenate(processed_train_data)
    processed_train_targets = np.concatenate(processed_train_targets)
    processed_test_data = np.concatenate(processed_test_data)
    true_rul = true_rul[0].values

    #Shuffle the data
    index = np.random.permutation(len(processed_train_targets))
    processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

    # Reshape
    processed_train_data = processed_train_data.reshape(-1, processed_train_data.shape[2])
    processed_test_data = processed_test_data.reshape(-1, processed_test_data.shape[2])

    # Construct the xgboost data structure
    dtrain = xgb.DMatrix(processed_train_data, label=processed_train_targets)
    dtest = xgb.DMatrix(processed_test_data)

    # Train the model
    num_rounds = 300
    params = {"max_depth": 3, "learning_rate": 1, "objective": "reg:squarederror"}
    bst = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=[(dtrain, "Train")], verbose_eval=50)

    #TODO: 详细再看下
    #Predict
    rul_pred = bst.predict(dtest)
    # First split predictions according to number of windows of each engine
    preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])

    mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
                                 for ruls_for_each_engine, num_windows in
                                 zip(preds_for_each_engine, num_test_windows_list)]
    RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
    print("RMSE: ", RMSE)