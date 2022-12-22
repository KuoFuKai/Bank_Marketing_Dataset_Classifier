import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, model_selection

import pandas as pd
import numpy as np

import requests
import io
import os

file_path = 'datasets/bank-additional-full-Final.csv'

bank_data = pd.read_csv(file_path)
bank_data.info()

bank_data = bank_data.apply(preprocessing.LabelEncoder().fit_transform)
print(bank_data)

X = bank_data.drop('y', axis=1)
y = bank_data['y']

mms = preprocessing.MinMaxScaler()  # X隨機值-最小值/最大-最小, 使資料介於0~1之間
bank_mms = mms.fit_transform(X)  # 資料會轉為Array型態
df_bank = pd.DataFrame(bank_mms,
                       columns=['age', 'job', 'marital', 'education', 'default', 'loan', 'contact', 'month', 'campaign',
                                'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                                'euribor3m', 'nr.employed'])  # 資料會變為Pandas表格型態

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_bank, y, test_size=0.3, random_state=42)

train_x = torch.tensor(X_train.to_numpy().astype(np.float32))
train_y = torch.tensor(y_train.to_numpy().astype(np.long), dtype=torch.long)
valid_x = torch.tensor(X_test.to_numpy().astype(np.float32))
valid_y = torch.tensor(y_test.to_numpy().astype(np.long), dtype=torch.long)


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_sample = len(x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_sample


train_set = dataset(train_x, train_y)

train_loader = DataLoader(dataset=train_set, batch_size=20, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(train_x.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )

    def forward(self, x):
        return self.net(x)


model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, )
epoch = 100
n_batch = len(train_loader)

for i in range(epoch):
    for j, (samples, labels) in enumerate(train_loader):
        pre = model(samples)
        labels = labels.view(-1)
        loss = criterion(pre, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch = {i + 1}/{epoch} batch = {j + 1}/{n_batch} loss = {loss:.4f}", end=' ')
        with torch.no_grad():
            pre = model(train_x)
            _, pre = torch.max(pre, 1)
            n_sample = len(train_x)
            n_correct = (train_y.view(-1) == pre).sum()
            print(f"train_acc = {n_correct / n_sample:.4f}")

with torch.no_grad():
    pre = model(valid_x)
    _, pre = torch.max(pre, 1)
    n_sample = len(valid_x)
    n_correct = (valid_y.view(-1) == pre).sum()
    print(f"valid_acc = {n_correct / n_sample}")
