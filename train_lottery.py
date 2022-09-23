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
  parser.add_argument('--epochs', default=20000, type=int) # 训练轮数
  parser.add_argument('--layers', default=3, type=int) # LSTM层数
  parser.add_argument('--input_size', default=6, type=int) #输入特征的维度
  parser.add_argument('--hidden_size', default=12, type=int) #隐藏层的维度
  parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
  parser.add_argument('--sequence_length', default=100, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
  parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
  parser.add_argument('--dropout', default=0.5, type=float)
  parser.add_argument('--save_file_red', default='model/lottery_red.pkl') # 模型保存位置
  parser.add_argument('--save_file_blue', default='model/lottery_blue.pkl') # 模型保存位置
  args = parser.parse_args()
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
  #device = torch.device("cuda" if args.useGPU else "cpu")
  args.device = device
  return args


class LotteryPredict:
  def __init__(self, args):
    self.model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=1 , output_size=6, dropout=args.dropout, batch_first=args.batch_first)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001
    self.criterion = nn.MSELoss()  # 定义损失函数
    self.corpusFile = args.corpusFile
    self.corpusFile = args.corpusFile
    self.sequence_length = args.sequence_length
    self.batch_size = args.batch_size
    self.save_file_red = args.save_file_red
    self.save_file_blue = args.save_file_blue
    self.device = args.device
    print("device : ",self.device)
    self.epochs = args.epochs
    self.useGPU = True

  def train(self):
    self.model.to(self.device)
    train_loader_red, test_loader_red, train_loader_blue, test_loader_blue = getData(self.corpusFile, self.sequence_length, self.batch_size)
    last_loss = 10
    for i in range(self.epochs):
      total_loss = 0
      count = 0
      for idx,(data, label) in enumerate(train_loader_red):
        if self.useGPU:
          data1 = data.squeeze(1).cuda()
          pred = self.model(Variable(data1).cuda())
          pred = pred[0,:]
          #print("Ethan pred : ", pred)
          #print("Ethan label: ",label)
          label = label.unsqueeze(1).cuda()

          # print(label.shape)
        else:
          data1 = data.squeeze(1)
          pred = self.model(Variable(data1))
          #pred = pred[1, :, :]
          label = label.unsqueeze(1)
        #print("Ethan shape pred: ",pred.size(),"label: " ,label[0, 0, :].size())
        loss = self.criterion(pred, label[0, 0, :])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
        count = count +1
      total_loss =  total_loss/count
      print("total loss : " , total_loss, "last loss: ",last_loss)
      if total_loss < last_loss:
        # torch.save(model, args.save_file)
        torch.save({'state_dict': self.model.state_dict()}, args.save_file_red)
        print('第%d epoch，保存模型' % i)
        last_loss = total_loss


if __name__ == "__main__":
    args = parse_arg()
    print(args)
    lottery = LotteryPredict(args)
    lottery.train()
    #for i, train_item in enumerate(train_loader):
    #  print("Ethan##: ", i, ": ", train_item)

    #load data
    

    #pred_obj = LotteryPredict(args)
    #pred_obj.train()