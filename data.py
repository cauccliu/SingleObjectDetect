""" 
pytorch-Single-target-detection
单目标检测：判断一张图片上是否有对应目标
    * 人脸检测
    * 小黄人检测

二分类：判断是否包含小黄人，输出层sigmoid（数据归一化为0-1，设置一个阈值来判断是否包含小黄人）
回归：画出框，输出层不需要激活(输出四个坐标值)
小黄人种类：多分类问题，输出层使用softmax（每类概率和为1）

数据集：包含正样本和负样本
数据集命名：1.0.0.0.0.0.0.jpg ： 

- 第一个数字为序号
- 第二个数字代表是否有小黄人，1有，0无
- 第三到第六个数字，代表位置。
- 第七位为种类数

处理数据
构建模型
train
test
模型评估
部署
"""


from torch.utils.data import Dataset
import os
# import cv2
import numpy as np
from PIL import Image
import torch

class MyDataset(Dataset):

    def __init__(self,root,is_train=True):
        self.dataset = []
        dir = 'train' if is_train else 'test'
        sub_dir = os.path.join(root,dir)
        img_list = os.listdir(sub_dir) # 获取文件夹下所有文件

        for i in img_list:
            img_dir = os.path.join(sub_dir,i)
            # print(img_dir)
            self.dataset.append(img_dir)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index): # 当数据被调用时就会触发该函数
        data = self.dataset[index]
        # img = cv2.imread(data) # 读到的图片是HWC ，卷积用到的是 NCHW，需要换轴 ，没有OpenCV用底下的两行代码操作代替

        # Load the image
        img = Image.open(data)
        img = img.resize((300,300))
        img = np.array(img)/255 # 读入图片并进行归一化
        
        #print(img.shape)
        # new_img = np.transpose(img,(2,0,1)) # HWC -> CHW (2,0,1) # Numpy做法 (3, 180, 298)
        new_img = torch.tensor(img).permute(2,0,1) # torch做法 torch.tensor([3, 180, 298])
        #print(new_img.shape)

        # 用点分割去取数据

        data_list = data.split('.')
        label = int(data_list[1])
        position = data_list[2:6]
        position = [int(i)/300 for i in position]
        sort = int(data_list[6])-1 # 0就变成-1了

        return np.float32(new_img),np.float32(label),np.float32(position),sort
        

if __name__ =='__main__':
    data = MyDataset('/home/liuchang/codetest/PythonCode/YellowPersonDetect/yellow_data',is_train=False)
    # data[0]
    for i in data:
        print(i)