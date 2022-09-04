from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
#from parser_my import args
from dataset import getData

import argparse
def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--corpusFile', default='data/lottery.csv')
  # TODO 常改动参数
  parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
  parser.add_argument('--epochs', default=100, type=int) # 训练轮数
  parser.add_argument('--layers', default=2, type=int) # LSTM层数
  parser.add_argument('--input_size', default=8, type=int) #输入特征的维度
  parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
  parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
  parser.add_argument('--sequence_length', default=5, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
  parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
  parser.add_argument('--dropout', default=0.1, type=float)
  parser.add_argument('--save_file', default='model/lottery.pkl') # 模型保存位置
  args = parser.parse_args()
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
  args.device = device
  return args


class LotteryPredict:
  def __init__(self, args):
    self.model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001
    self.criterion = nn.MSELoss()  # 定义损失函数
    self.corpusFile = args.corpusFile
    self.corpusFile = args.corpusFile
    self.sequence_length = args.sequence_length
    self.batch_size = args.batch_size
    self.save_file = args.save_file
    self.device = args.device

  def train(self):
    self.model.to(self.device)
    train_loader, test_loader = getData(self.corpusFile, self.sequence_length, self.batch_size)
    for i in range(self.epochs):
      total_loss = 0
      for idx, in enumerate(train_loader):
        if self.useGPU:
          data1 = data.squeeze(1).cuda()
          pred = self.model(Variable(data1).cuda())
          # print(pred.shape)
          pred = pred[1,:,:]
          label = label.unsqueeze(1).cuda()
          # print(label.shape)
        else:
          data1 = data.squeeze(1)
          pred = self.model(Variable(data1))
          pred = pred[1, :, :]
          label = label.unsqueeze(1)
        loss = self.criterion(pred, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
      print(total_loss)
      if i % 10 == 0:
        # torch.save(model, args.save_file)
        torch.save({'state_dict': self.model.state_dict()}, args.save_file)
        print('第%d epoch，保存模型' % i)
    # torch.save(model, args.save_file)
    torch.save({'state_dict': self.model.state_dict()}, args.save_file)'''

if __name__ == "__main__":
    args = parse_arg()
    print(args)
    pred_obj = LotteryPredict(args)
    pred_obj.train()