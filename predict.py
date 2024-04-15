import os
import cv2

import torch

from net import MyNet

if __name__ == '__main__':
    img_name = os.listdir('/home/liuchang/codetest/PythonCode/YellowPersonDetect/yellow_data/test') # 获取文件路径
    for i in img_name:
        img_dir = os.path.join('/home/liuchang/codetest/PythonCode/YellowPersonDetect/yellow_data/test',i) # 获取img

        img = cv2.imread(img_dir)

        position = i.split('.')[2:6] # 读取位置坐标
        position = [int(j) for j in position]
        sort =  i.split('.')[6]
        cv2.rectangle(img,(position[0],position[1]),(position[3],position[4]),(0,255,0),thickness=2) # 画label框
        cv2.putText(img,sort,(position[0],position[1]-3),cv2.FONT_HERSHEY_SIMPLEX,2,1,(0,255,0),thickness=2) #左上角的点上写出类型

        model = MyNet()
        model.load_state_dict(torch.load('param/2024-04-10-14_37_43.382495-50.pt'))
        new_img = torch.tensor(img).permute(2,0,1)
        torch.unsqueeze(new_img,dim=0)/255 # 传递一个维度，归一化

        label_out,out_position,out_sort = model(new_img)
        label_out = torch.sigmoid(label_out) # 网络模型没有做归一化，这里就要做归一化
        out_sort = torch.argmax(torch.softmax(out_sort,dim=1))

        out_position = out_position[0]*300
        out_position = [int(i) for i in out_position]
        if label_out.item()>0.5 :
            cv2.rectangle(img,(out_position[0],out_position[1]),(out_position[3],out_position[4]),(255,0,0),thickness=2) # 画label框
            cv2.putText(img,str(out_sort.item()),(out_position[0],out_position[1]-3),cv2.FONT_HERSHEY_SIMPLEX,2,1,(255,0,0),thickness=2) #左上角的点上写出类型
        
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.destroyAllWindows() # 展示图片后关闭图片


