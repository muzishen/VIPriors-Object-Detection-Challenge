import os
import moxing as mox
from naie.context import Context

from naie.datasets import get_data_reference
from naie.feature_processing import data_flow

import json
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
import shutil
import glob
#from convert2coco import convert_main

#download dataset
data_reference = get_data_reference(dataset="custom_city", dataset_entity="custom_city")
file_paths = data_reference.get_files_paths()
print(file_paths)
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/524f086e8fd44fd98b09634d0228103b/Dataset/custom_city/custom_city/data/custom_city.zip', '/cache/custom_city.zip')

os.system('unzip /cache/custom_city.zip -d /cache')
os.system('ls /cache/')
os.system('ls /cache/custom_city')

# os.mkdir('/cache/coco_voc')

# os.mkdir('/cache/xray_voc/images')
# os.mkdir('/cache/xray_voc/xmls')

# for file_name in glob.glob('/cache/xray/train/domain1/*.jpg'):
#     filename = file_name.split('/')[-1]
#     shutil.copy(file_name, '/cache/xray_voc/images/'+filename)
#     shutil.copy(os.path.join('/cache/xray/train/domain1/XML', filename.replace('jpg', 'xml')), '/cache/xray_voc/xmls/'+filename.replace('jpg', 'xml'))

# for file_name in glob.glob('/cache/xray/train/domain2/*.jpg'):
#     filename = file_name.split('/')[-1]
#     shutil.copy(file_name, '/cache/xray_voc/images/'+filename)
#     shutil.copy(os.path.join('/cache/xray/train/domain2/XML', filename.replace('jpg', 'xml')), '/cache/xray_voc/xmls/'+filename.replace('jpg', 'xml'))


# for file_name in glob.glob('/cache/xray/train/domain3/*.jpg'):
#     filename = file_name.split('/')[-1]
#     shutil.copy(file_name, '/cache/xray_voc/images/'+filename)
#     shutil.copy(os.path.join('/cache/xray/train/domain3/XML', filename.replace('jpg', 'xml')), '/cache/xray_voc/xmls/'+filename.replace('jpg', 'xml'))



# for file_name in glob.glob('/cache/xray/train/domain4/*.jpg'):
#     filename = file_name.split('/')[-1]
#     shutil.copy(file_name, '/cache/xray_voc/images/'+filename)
#     shutil.copy(os.path.join('/cache/xray/train/domain4/XML', filename.replace('jpg', 'xml')), '/cache/xray_voc/xmls/'+filename.replace('jpg', 'xml'))


# for file_name in glob.glob('/cache/xray/train/domain5/*.jpg'):
#     filename = file_name.split('/')[-1]
#     shutil.copy(file_name, '/cache/xray_voc/images/'+filename)
#     shutil.copy(os.path.join('/cache/xray/train/domain5/XML', filename.replace('jpg', 'xml')), '/cache/xray_voc/xmls/'+filename.replace('jpg', 'xml'))


# for file_name in glob.glob('/cache/xray/train/domain6/*.jpg'):
#     filename = file_name.split('/')[-1]
#     shutil.copy(file_name, '/cache/xray_voc/images/'+filename)
#     shutil.copy(os.path.join('/cache/xray/train/domain6/XML', filename.replace('jpg', 'xml')), '/cache/xray_voc/xmls/'+filename.replace('jpg', 'xml'))


# os.mkdir('/cache/xray_dataset')




# os.mkdir('/cache/xray_dataset/domain1')
# os.mkdir('/cache/xray_dataset/domain2')
# os.mkdir('/cache/xray_dataset/domain3')
# os.mkdir('/cache/xray_dataset/domain4')
# os.mkdir('/cache/xray_dataset/domain5')
# os.mkdir('/cache/xray_dataset/domain6')



#convert_main('/cache/xray_voc/xmls', '/cache/xray_voc/images', '/cache/xray_dataset', split_rate=0.8)
# convert_main('/cache/xray/train/domain2/XML', '/cache/xray/train/domain2', '/cache/xray_dataset/domain2', split_rate=0.8)
# convert_main('/cache/xray/train/domain3/XML', '/cache/xray/train/domain3', '/cache/xray_dataset/domain3', split_rate=0.8)
# convert_main('/cache/xray/train/domain4/XML', '/cache/xray/train/domain4', '/cache/xray_dataset/domain4', split_rate=0.8)
# convert_main('/cache/xray/train/domain5/XML', '/cache/xray/train/domain5', '/cache/xray_dataset/domain5', split_rate=0.8)
# convert_main('/cache/xray/train/domain6/XML', '/cache/xray/train/domain6', '/cache/xray_dataset/domain6', split_rate=0.8)



#c_r2_101
data_reference = get_data_reference(dataset="c_r2_101", dataset_entity="c_r2_101")
file_paths = data_reference.get_files_paths()
print(file_paths)
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/524f086e8fd44fd98b09634d0228103b/Dataset/c_r2_101/c_r2_101/data/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth', '/cache/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth')
# os.system('tar -xvf /cache/c_r2_101.tar.gz -C /cache')
# os.system('ls /cache')

# # download premodel
# #c_r101
# data_reference = get_data_reference(dataset="c_r101", dataset_entity="c_r101")
# file_paths = data_reference.get_files_paths()
# print(file_paths)
# mox.file.copy('s3://bucket-6ck6p6db/06292eb2cb8010dd1f2dc000a17f2c89/e379519b18074acea090d9c2d95b1635/Dataset/c_r101/c_r101/c_r101/c_r101.tar.gz', '/cache/c_r101.tar.gz')

# os.system('tar -xvf /cache/c_r101.tar.gz -C /cache')

# os.system('python download_hh.py')

hw_model_path = Context.get_model_path()
print(hw_model_path)

# os.system('python download_hw.py')
# os.chdir('./cocoapi')
#os.system('pip install pycocotools/')
os.chdir('./mmdection')
os.system('pwd')
os.system('ls')
os.system('pip install pycocotools/')
os.system('python setup.py develop')
#os.system('ls /cache/xray_dataset/annotations')
#os.system('python tools/train.py custom_configs/c_r2net_sf.py')
#os.system('python tools/train.py custom_configs/cascade_rcnn_r2_101_fpn_20e_coco_xray.py')
#os.system('python tools/train.py custom_configs/cascade_rcnn_r101_fpn_20e_coco_xray.py')
os.system('python tools/train.py custom_configs/cascade_rcnn_r2_101_custom_city.py')
#os.system('python tools/train.py custom_configs/eccv_coco_detectors.py')
#os.system('python -W ignore tools/train.py custom_configs/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py')
#os.system('./tools/dist_train.sh custom_configs/cascade_rcnn_db_r101_fpn_20e_coco.py 8')


mox.file.copy('/cache/log/epoch_12.pth', os.path.join(hw_model_path,  'epoch_12.pth'))
mox.file.copy('/cache/log/epoch_24.pth', os.path.join(hw_model_path,  'epoch_24.pth'))
# mox.file.copy('/cache/log/epoch_36.pth', os.path.join(hw_model_path,  'epoch_36.pth'))
# mox.file.copy('/cache/log/epoch_48.pth', os.path.join(hw_model_path,  'epoch_48.pth'))
# mox.file.copy('/cache/log/epoch_60.pth', os.path.join(hw_model_path,  'epoch_60.pth'))
# mox.file.copy('/cache/log/epoch_72.pth', os.path.join(hw_model_path,  'epoch_72.pth'))
# mox.file.copy('/cache/log/epoch_73.pth', os.path.join(hw_model_path,  'epoch_73.pth'))

# mox.file.copy('/cache/log/epoch_84.pth', os.path.join(hw_model_path,  'epoch_84.pth'))
# mox.file.copy('/cache/log/epoch_96.pth', os.path.join(hw_model_path,  'epoch_96.pth'))
# mox.file.copy('/cache/log/epoch_108.pth', os.path.join(hw_model_path,  'epoch_108.pth'))
# mox.file.copy('/cache/log/epoch_120.pth', os.path.join(hw_model_path,  'epoch_120.pth'))
# mox.file.copy('/cache/log/epoch_132.pth', os.path.join(hw_model_path,  'epoch_132.pth'))
# mox.file.copy('/cache/log/epoch_144.pth', os.path.join(hw_model_path,  'epoch_144.pth'))
# mox.file.copy('/cache/log/epoch_145.pth', os.path.join(hw_model_path,  'epoch_145.pth'))
# # mox.file.copy('/cache/log/epoch_21.pth', os.path.join(hw_model_path,  'epoch_21.pth'))
# # mox.file.copy('/cache/log/epoch_22.pth', os.path.join(hw_model_path,  'epoch_22.pth'))
# # mox.file.copy('/cache/log/epoch_23.pth', os.path.join(hw_model_path,  'epoch_23.pth'))
# # mox.file.copy('/cache/log/epoch_24.pth', os.path.join(hw_model_path,  'epoch_24.pth'))

