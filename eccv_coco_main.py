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
data_reference = get_data_reference(dataset="eccv_coco1", dataset_entity="eccv_coco1")
file_paths = data_reference.get_files_paths()
print(file_paths)
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/524f086e8fd44fd98b09634d0228103b/Dataset/eccv_coco1/eccv_coco1/data/eccv_coco.zip', '/cache/eccv_coco.zip')

os.system('unzip /cache/eccv_coco.zip -d /cache')
os.system('ls /cache/')

hw_model_path = Context.get_model_path()
print(hw_model_path)

# os.system('python download_hw.py')
# os.chdir('./cocoapi')
os.system('pip install pycocotools/')
os.chdir('./mmdection')
os.system('pwd')
os.system('ls')
os.system('pip install pycocotools/')
os.system('python setup.py develop')
#os.system('ls /cache/xray_dataset/annotations')
#os.system('python tools/train.py custom_configs/c_r2net_sf.py')
#os.system('python tools/train.py custom_configs/cascade_rcnn_r2_101_fpn_20e_coco_xray.py')
#os.system('python tools/train.py custom_configs/cascade_rcnn_r101_fpn_20e_coco_xray.py')
os.system('python tools/train.py custom_configs/eccv_coco_cascade_r2_50.py')
#os.system('python tools/train.py custom_configs/eccv_coco_detectors.py')
#os.system('python -W ignore tools/train.py custom_configs/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py')
#os.system('./tools/dist_train.sh custom_configs/cascade_rcnn_db_r101_fpn_20e_coco.py 8')


mox.file.copy('/cache/log/epoch_12.pth', os.path.join(hw_model_path,  'epoch_12.pth'))
mox.file.copy('/cache/log/epoch_24.pth', os.path.join(hw_model_path,  'epoch_24.pth'))
mox.file.copy('/cache/log/epoch_36.pth', os.path.join(hw_model_path,  'epoch_36.pth'))
mox.file.copy('/cache/log/epoch_48.pth', os.path.join(hw_model_path,  'epoch_48.pth'))
mox.file.copy('/cache/log/epoch_60.pth', os.path.join(hw_model_path,  'epoch_60.pth'))
mox.file.copy('/cache/log/epoch_72.pth', os.path.join(hw_model_path,  'epoch_72.pth'))
mox.file.copy('/cache/log/epoch_73.pth', os.path.join(hw_model_path,  'epoch_73.pth'))

mox.file.copy('/cache/log/epoch_84.pth', os.path.join(hw_model_path,  'epoch_84.pth'))
mox.file.copy('/cache/log/epoch_96.pth', os.path.join(hw_model_path,  'epoch_96.pth'))
mox.file.copy('/cache/log/epoch_108.pth', os.path.join(hw_model_path,  'epoch_108.pth'))
mox.file.copy('/cache/log/epoch_120.pth', os.path.join(hw_model_path,  'epoch_120.pth'))
mox.file.copy('/cache/log/epoch_132.pth', os.path.join(hw_model_path,  'epoch_132.pth'))
mox.file.copy('/cache/log/epoch_144.pth', os.path.join(hw_model_path,  'epoch_144.pth'))
mox.file.copy('/cache/log/epoch_145.pth', os.path.join(hw_model_path,  'epoch_145.pth'))



