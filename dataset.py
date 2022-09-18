import pandas
from pandas import read_csv
from pandas import Series
from pandas import DataFrame
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch
from torchvision import transforms
from parser_my import args


def getData(corpusFile,sequence_length,batchSize):
    # 数据预处理 ，去除id、股票代码、前一天的收盘价、交易日期等对训练无用的无效数据
    stock_data = read_csv(corpusFile)
    stock_data.drop('id', axis=1, inplace=True)  # 删除第一列’id‘
    stock_data.drop('second', axis=1, inplace=True)  # 删除列’pre_close‘
    stock_data.drop('first', axis=1, inplace=True)  # 删除列’pre_close‘
    stock_data.drop('time', axis=1, inplace=True)  # 删除列’trade_date‘
    stock_data.drop('money', axis=1, inplace=True)  # 删除列’trade_date‘
    #print(stock_data)
    #lottery_data = [int(x) for x in stock_data["red_num"]]

    #money_max = stock_data['money'].max() #投注的最大值
    #money_min = stock_data['money'].min() #投注的最小值
    #money_diff = money_max - money_min
    #money = stock_data['money'].apply(lambda x: (x - money_min)/money_diff )  # min-max标准化
    red_num = stock_data['boll_num'].apply(lambda x:  Series([float(x)/33 for x in x.split()[:-1]], dtype=float))#str
    blue_num = stock_data['boll_num'].apply(lambda x:  Series([float(x)/16 for x in x.split()[-1:]], dtype=float))#str
    sequence = sequence_length
    red = []
    red_y = []
    blue = []
    blue_y = []
    for i in range(red_num.shape[0]-sequence):
        red.append(np.array(red_num.iloc[i:(i+sequence)].values, dtype=np.float32))
        red_y.append(np.array(red_num.iloc[i+sequence].values, dtype=np.float32))
        blue.append(np.array(blue_num.iloc[i:(i + sequence)].values, dtype=np.float32))
        blue_y.append(np.array(blue_num.iloc[i + sequence].values, dtype=np.float32))

    # 构建batch
    total_len = len(red)
    # print(total_len)
    #print("Ethan red : ", red)
    #print("Ethan blue: ", blue)

    train_red, train_red_y = np.array(red[:int(0.99*total_len)]), np.array(red[:int(0.99*total_len)])
    test_red, test_red_y = np.array(red[(int(0.99*total_len)):]), np.array(red[int(0.99*total_len):])
    train_blue, train_blue_y = np.array(blue[:int(0.99*total_len)]), np.array(blue[:int(0.99*total_len)])
    test_blue, test_blue_y = np.array(blue[(int(0.99*total_len)):]), np.array(blue[int(0.99*total_len):])
    train_red_dataset = Mydataset(train_red, train_red_y)
    test_red_dataset = Mydataset(test_red, test_red_y)
    train_blue_dataset = Mydataset(train_blue, train_blue_y)
    test_blue_dataset = Mydataset(test_blue, test_blue_y)
    train_loader_red = DataLoader(train_red_dataset, batch_size=batchSize, shuffle=True)
    test_loader_red = DataLoader(test_red_dataset, batch_size=batchSize, shuffle=True)
    train_loader_blue = DataLoader(train_blue_dataset, batch_size=batchSize, shuffle=True)
    test_loader_blue = DataLoader(test_blue_dataset, batch_size=batchSize, shuffle=True)
    return train_loader_red, test_loader_red, train_loader_blue, test_loader_blue


class Mydataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.tranform = transform

    def __getitem__(self, index):
        print("index:",index)
        if index >= 0 :
          x = self.x[index]
          y = self.y[index]
          #print("x shape ",x.shape)
          #print("y shape ",y.shape)

          if self.tranform:
              return self.tranform(x), y
          return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        if len(self.x) ==len(self.y):
            return len(self.x)
        else:
            return -1


def load_data(file_name):
    df = pd.read_csv('data/new_data/' + file_name, encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_us(B):
    print('data processing...')
    dataset = load_data()
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size, shuffle):
        load = data[data.columns[1]]
        load = load.tolist()
        data = data.values.tolist()
        load = (load - n) / (m - n)
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, True)
    Val = process(val, B, True)
    Dte = process(test, B, False)

    return Dtr, Val, Dte, m, n