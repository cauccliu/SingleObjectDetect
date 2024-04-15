from net import MyNet
from data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn,optim
import torch
import os
import datetime

DEVICE='cuda'

class Train:
    def __init__(self,root,weight_path):
        self.summaryWriter = SummaryWriter('logs')
        self.train_dataset = MyDataset(root=root,is_train= True)
        self.test_dataset = MyDataset(root=root,is_train= False)
        # 挺好，现在知道这玩意是在干嘛了
        self.train_dataLoader = DataLoader(self.train_dataset,batch_size=50,shuffle=True)
        self.test_dataLoader = DataLoader(self.test_dataset,batch_size=50,shuffle=True)

        self.net = MyNet().to(DEVICE) # 放到gpu上

        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))

        self.opt = optim.Adam(self.net.parameters())

        # 分类，回归，都有损失，怎么计算？
        
        self.label_loss_fun = nn.BCEWithLogitsLoss()  # 二分类用BCELOGIST，里面自带着sigmod激活
        self.position_loss_fun = nn.MSELoss() # 回归，MSE LOSS
        self.sort_loss_fun = nn.CrossEntropyLoss() # 多分类：cross LOSS，多指交叉熵，自带softmax激活

        self.train = True
        self.test = True
        
    def __call__(self):
        index1,index2 = 0,0
        for epoch in range(1000): # 训练1000个epoch
            if self.train:
                for i, (img,label,position,sort) in enumerate(self.train_dataLoader):
                    self.net.train()
                    
                    img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE) # 数据放到GPU上
                    # print(img.shape)
                    # print(label.shape)
                    out_label,out_position,out_sort = self.net(img)
                    # print(out_label,out_position,out_sort) 
                    # print(out_label.shape)
                    
                    label_loss = self.label_loss_fun(out_label,label)
                    # print(label_loss)

                    position_loss=self.position_loss_fun(out_position,position)
                    # print(position.shape)
                    # print(out_position.shape)
                    # print(position_loss)
                    
                    sort = sort[torch.where(sort >= 0)]
                    out_sort = out_sort[torch.where(sort >= 0)]
                    sort_loss = self.sort_loss_fun(out_sort,sort)
                    # print(sort_loss)

                    train_loss = label_loss + position_loss + sort_loss
                    
                    self.opt.zero_grad()
                    train_loss.backward()
                    self.opt.step()

                    if i%10 ==0 :
                        print(f'train_loss{i}===>',train_loss.item())
                        self.summaryWriter.add_scalar('train_loss',train_loss,index1)
                        index1 +=1
                    
                data_time = str(datetime.datetime.now()).replace(' ', '-').replace(':','_').replace('·','_')
                save_dir = 'param'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存权重文件
                torch.save(self.net.state_dict(), f'{save_dir}/{data_time}-{epoch}.pt')
            
            if self.test:
                sum_sort_acc,sum_label_acc = 0,0
                for i, (img,label,position,sort) in enumerate(self.test_dataLoader):
                    self.net.train()
                    
                    img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE) # 数据放到GPU上
                    
                    out_label,out_position,out_sort = self.net(img)
                    
                    label_loss = self.label_loss_fun(out_label,label)

                    position_loss=self.position_loss_fun(out_position,position)
                    
                    # print(position_loss)
                    
                    sort = sort[torch.where(sort>=0)]
                    out_sort = out_sort[torch.where(sort >= 0)]
                    sort_loss = self.sort_loss_fun(out_sort,sort)
                    # print(sort_loss)

                    test_loss = label_loss + position_loss + sort_loss

                    out_label = torch.tensor(torch.sigmoid(out_label))
                    out_label[torch.where(out_label>=0.5)] = 1
                    out_label[torch.where(out_label<0.5)] = 0

                    label_acc = torch.mean(torch.eq(out_label,label).float())
                    sum_label_acc += label_acc 

                    # 求准确率
                    # out_sort = torch.argmax(torch.softmax(out_sort,dim=1))
                    if out_sort.numel() > 0:
                        out_sort = torch.argmax(torch.softmax(out_sort, dim=1))
                        out_sort = out_sort.to(sort.device)  # Move out_sort to the same device as sort
                    else:
                        out_sort = torch.tensor([], device=sort.device)  # Or handle the empty case 

                    sort_acc = torch.mean(torch.eq(sort,out_sort).float())
                    sum_sort_acc += sort_acc
                    
                    if i%10 ==0 :
                        print(f'test_loss{i}===>',test_loss.item())
                        self.summaryWriter.add_scalar('test_loss',test_loss,index2)
                        index2 +=1
                    
                avg_sort_acc = sum_sort_acc/i
                print(f'avg_sort_acc {epoch}====>',avg_sort_acc)
                self.summaryWriter.add_scalar('avg_sort_acc',avg_sort_acc,epoch)

                avg_label_acc = sum_label_acc/i
                print(f'avg_label_acc {epoch}====>',avg_label_acc)
                self.summaryWriter.add_scalar('avg_label_acc',avg_label_acc,epoch) 

if __name__ =='__main__':
    train = Train('/home/liuchang/codetest/PythonCode/YellowPersonDetect/yellow_data','2024-04-10-14_26_21.255621-9.pt')
    train()