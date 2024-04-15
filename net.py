from torch import nn 
import torch

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,11,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(11,22,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(22,32,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,3),
            nn.LeakyReLU(),
        )

        # label的预测值
        self.label_layers = nn.Sequential(
            nn.Conv2d(128,1,19),
            nn.LeakyReLU()
        )

        # position的预测值
        self.position_layer = nn.Sequential(
            nn.Conv2d(128,4,19),
            nn.LeakyReLU()
        )

        # sort的预测值：现在的模拟数据类别只有3类，所以这里是3，真实的数据集有20类，这里要改。
        self.sort_layer = nn.Sequential(
            nn.Conv2d(128,20,19),
            nn.LeakyReLU()
        )

    def forward(self,x):
        out = self.layers(x)

        label = self.label_layers(out)
        # 降维，从四维降为二维
        label = torch.squeeze(label,dim=2)
        label = torch.squeeze(label,dim=2)
        label = torch.squeeze(label,dim=1)

        position = self.position_layer(out)
        position = torch.squeeze(position,dim=2)
        position = torch.squeeze(position,dim=2)

        sort = self.sort_layer(out)
        sort = torch.squeeze(sort,dim=2)
        sort = torch.squeeze(sort,dim=2)

        return label,position,sort

if __name__ =='__main__':
    net = MyNet()
    x = torch.randn(3,3,300,300)

    print(net(x)[0].shape)
    print(net(x)[1].shape)
    print(net(x)[2].shape)