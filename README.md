### [A Report to VIPriors Object Detection Challenge](https://arxiv.org/pdf/2104.09059.pdf)
 This is 1st Code for the VIPriors Object Detection of 2020 ECCV Workshop
 VIPriors workshop 的完整代码
#### 比赛规则：禁止使用预训练模型！！！

#### 1. 离线增强
###### (1) 复制所有图片6次，并进行随机自动增强(策略包括，亮度，洗牌通道，对比度，噪声)
###### (2) 对类别较少的样本使用bbox增强,本部分参考开源仓库[bbox-augmentation](https://github.com/mukopikmin/bounding-box-augmentation)
###### (3) 对以上所有数据应用albumentations增强库(策略只包括 色彩饱和度和中值模糊)

#### 2. 在线增强
##### (1) bbox-jitter，检测框抖动 本部分代码参考开源仓库[bbox-jitter](https://github.com/cizhenshi)
##### (2) grid-mask 擦除
##### (3) mix-up 混合

#### 3. 训练技巧
##### (1) 增加 global context feature
##### (2) 在骨干网中加入switchable atrous convolution 
##### (3) 根据kaiming大神的论文，把骨干网的BN换成GN，效果神奇 
##### (4) 使用SGD_GC优化器，本部分参考开源仓库[Gradient-Centralization](https://github.com/Yonghongwei/Gradient-Centralization)

#### 使用了Cascade-RCNN级联检测器，骨干网 ResNeSt-152, Res2net-101, SeNet-154,最后结果检测框merge 

##### ![实验结果](/img/20201221125932.png)

### 3个模型的配置文件路径：
 [/mmdection/custom_configs/](/mmdection/custom_configs/)
## Step1：运行方法 
```python eccv_coco_main.py```
训练得到3个模型
## Step2:推理部分
```python eccv_coco_test.py```
推理得到3个结果
## setep3:融合
```python inference_merge.py```

#### 感谢opne-mmlab实验室开源仓库[mmdetection](https://github.com/open-mmlab/mmdetection)！！！
