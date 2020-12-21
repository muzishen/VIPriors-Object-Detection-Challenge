# A Report to VIPriors Object Detection Challenge
 This is 1st Code for the VIPriors Object Detection of 2020 ECCV Workshop
 VIPriors workshop 的完整代码
##
比赛规则：禁止使用预训练模型！！！
## 感谢opne-mmlab实验室开源仓库[mmdetection](https://github.com/open-mmlab/mmdetection)
## 离线增强
#### (1) 复制所有图片6次，并进行随机自动增强(策略包括，亮度，洗牌通道，对比度，噪声)
#### (2) 对类别较少的样本使用bbox增强,本部分参考开源仓库[bbox-augmentation](https://github.com/mukopikmin/bounding-box-augmentation)
#### (3) 对以上所有数据应用albumentations增强库(策略只包括 色彩饱和度和中值模糊)

## 在线增强
#### bbox-jitter，本部分代码参考开源仓库[bbox-jitter](https://github.com/cizhenshi)
#### grid-mask
#### mix-up

### 增强 global context feature
### 使用switchable atrous convolution to the backbone
### 根据kaiming大神的论文，把骨干网的BN换成GN，效果神奇 
### 使用SGD_GC优化器，本部分参考开源仓库[Gradient-Centralization](https://github.com/Yonghongwei/Gradient-Centralization)

