# 2024CAIP
算法调优比赛赛道-甲骨文分类

用于记录本队（其实也只有我一个人qwq）参加此次比赛的全过程

官方给的数据集在这链接：https://pan.baidu.com/s/1icPaGlZfs-0mfVxxBMIiQg?pwd=xaaf 
提取码：xaaf 

我自行训练的分类模型在这：https://pan.baidu.com/s/1jC4gcGBLUkhncU5DeyrGSw?pwd=j86n 
提取码：j86n 


## 数据集
该比赛所使用的数据集来自“殷契文渊”平台提供的 HWOBC 数据集，“殷契文渊”平台是世界上现有的资料最齐全、最规范、最权威的甲骨文数据平台，对全世界免费开放，改变了过去甲骨文研究的“窘境”。  
经过筛选保留了 376 种已破译的甲骨文字符，其中每个类别平均包含 100 张以上的字符图片，分为训练集和测试集。

- 数据展示


 
<div class='insertContainerBox row'>
<div class='insertItem' align=center><img src="https://imgbed.momodel.cn/20231202093532.png" width="40px"/></div>
   <div class='insertItem' align=center><img src="https://imgbed.momodel.cn/20231202093548.png" width="40px"/></div> 
       <div class='insertItem' align=center><img src="https://imgbed.momodel.cn/20231202093607.png" width="40px"/></div> 
    <div class='insertItem' align=center><img src="https://imgbed.momodel.cn/20231202093630.png" width="40px"/></div> 
</div>

![image](https://github.com/user-attachments/assets/56dede9c-c68f-4211-892a-3ddb171b1615)




# 文件说明
## main.ipynb
官方提供的说明文档，包含数据集的构成、来源等信息，同时说明了答题格式

## Resnet_train.py
Resnet网络的训练代码

## GoogleNet_train.py
GoogleNet网络的训练代码

## 模型测试.ipynb
用于测试代码能否对数据集进行推理与输出结果，已经调试成适合Mo平台输入输出的格式

## model.zip
存放了训练出来的模型



