import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as py
import pandas as pd

# 从excel中读取数据
class ExcelDataset(Dataset):
    def __init__(self, filepath="./data/data.xlsx",sheet_name=0):
        print(f"reading {filepath}, sheet={sheet_name}")
        df = pd.read_excel(
            filepath, header=0, index_col=0,
            names=['feat1','feat2','label'],
            sheet_name=sheet_name,
            dtype={"feat1":np.float32, "feat2":np.float32, "label":np.int32}
        )
        print(f"the shape of dataframe is {df.shape}")


    def __len__(self):

