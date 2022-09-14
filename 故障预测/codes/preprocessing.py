import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


"""
    Preprocessing
"""
# Function that automatically finds RUL values for both models if it is supplied with the total number of cycles.
def process_targets(data_length, early_rul = None):
    """
    Takes datalength and earlyrul as input and
    creates target rul.
    """
    if early_rul == None:
        return np.arange(data_length-1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length-1, -1, -1)
        else:
            return np.append(early_rul*np.ones(shape = (early_rul_duration,)), np.arange(early_rul-1, -1, -1))


# Separates input data & corresponding target value into batches * window_length * shift
def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length,
                                                                                                num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


# Prepares a number of test examples
# @TODO:下面还有点没看明白，日后再复习，还有上面的函数
def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    """
    :param test_data_for_an_engine: input test data
    :param window_length: window length of data
    :param shift: overlap between data
    :param num_test_windows: number of examples to take
    :return:
    """
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:  # 只要第一组
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        extracted_num_test_windows = max_num_test_batches
        return batched_test_data_for_an_engine, extracted_num_test_windows
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        extracted_num_test_windows = num_test_windows
        return batched_test_data_for_an_engine, extracted_num_test_windows


"""
    test above functions
"""
def test():
    # Checks generating batches of train data and their target
    input1 = np.arange(40).reshape(10, 4)  # 4 records and 5 features
    window_length = 5
    shift = 5
    generated_train_data = process_input_data_with_targets(input_data=input1, window_length=window_length,
                                                                        shift=shift)
    print("Check generated train data", "\n", generated_train_data, '\n')

    # Checks generating batches of test data
    check_data = np.reshape(np.arange(24), newshape=(6, 4))
    last_examples, num_last_examples = process_test_data(check_data, window_length=2, shift=1,
                                                                      num_test_windows=3)
    print("Check generated test data", "\n", last_examples)
    print()
    print(num_last_examples)

    """
        After being processed by above two functions, we can get any kinds of data
    """
