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

#
def getData(corpusFile,sequence_length,batchSize):
    # 数据预处理 ，去除id、股票代码、前一天的收盘价、交易日期等对训练无用的无效数据
    stock_data = read_csv(corpusFile)
    stock_data.drop('id', axis=1, inplace=True)  # 删除第一列’id‘
    stock_data.drop('second', axis=1, inplace=True)  # 删除列’pre_close‘
    stock_data.drop('time', axis=1, inplace=True)  # 删除列’trade_date‘
    #print(stock_data)
    #lottery_data = [int(x) for x in stock_data["red_num"]]

    money_max = stock_data['money'].max() #投注的最大值
    money_min = stock_data['money'].min() #投注的最小值
    money_diff = money_max - money_min
    money = stock_data['money'].apply(lambda x: (x - money_min)/money_diff )  # min-max标准化
    red_num = stock_data['boll_num'].apply(lambda x:  Series([float(x)/33 for x in x.split()[:-1]],dtype=float))#str
    blue_num= stock_data['boll_num'].apply(lambda x:  Series([float(x)/16 for x in x.split()[-1:]],dtype=float))#str
    #print(money)
    #print(red_num)
    #print(blue_num)
    sequence = sequence_length
    # 构建batch
    total_len = money.shape[0]
    # print(total_len)

    train_red, train_blue, train_money = red_num, blue_num , money
    test_red, test_blue,test_money = red_num[-101:], blue_num[-101:], money[-101:]
    train_dataset = Mydataset(train_red, train_blue,train_money)
    test_dataset = Mydataset(test_red, test_blue, test_money)
    train_loader = DataLoader(train_dataset, batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True)
    return train_loader, test_loader


class Mydataset(Dataset):
    def __init__(self, red, blue, money, transform=None):
        self.red = red
        self.blue = blue
        self.money = money
        self.tranform = transform

    def __getitem__(self, index):
        red = self.red.iloc[index]
        blue = self.blue.iloc[index]
        money = self.money.iloc[index]
        if self.tranform != None:
            return self.tranform(red), blue, money
        print(red, blue, money)
        return torch.tensor([red, blue, money])

    def __len__(self):
        return len(self.red)


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