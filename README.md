# A Report to VIPriors Object Detection Challenge
 This is 1st Code for the VIPriors Object Detection of 2020 ECCV Workshop
 
 ![image](https://github.com/muzishen/A-Report-to-VIPriors-Object-Detection-Challenge/blob/master/demo3.png)
## Offline Data Augmentation
### (1)  Copy a single picture 6 times and random auto augmentation (e.g., brightness, shuffle channel, contrast, noise) 
### (2) Use bounding-box augmentation to random crop classes with few samples 
### (3) Apply albumentations library  to above all data (e.g., hue saturation value, median blur) 
## Online Data Augmentation
![image](https://github.com/muzishen/A-Report-to-VIPriors-Object-Detection-Challenge/blob/master/demo.png)
### bbox-jitter, grid-mask, and mix-up.
## Embed global context feature
## Use switchable atrous convolution to the backbone
## Replace all batch normalization with group normalization
## Apply gradient centralization 
![image](https://github.com/muzishen/A-Report-to-VIPriors-Object-Detection-Challenge/blob/master/demo2.png)
