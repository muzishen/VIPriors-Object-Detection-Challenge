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

data_reference = get_data_reference(dataset="eccv_coco1", dataset_entity="eccv_coco1")
file_paths = data_reference.get_files_paths()
print(file_paths)
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/524f086e8fd44fd98b09634d0228103b/Dataset/eccv_coco1/eccv_coco1/data/eccv_coco.zip', '/cache/eccv_coco.zip')

os.system('unzip /cache/eccv_coco.zip -d /cache')
os.system('ls /cache/')

# #c_r2_101
# data_reference = get_data_reference(dataset="c_r2_101", dataset_entity="c_r2_101")
# file_paths = data_reference.get_files_paths()
# print(file_paths)
# mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/524f086e8fd44fd98b09634d0228103b/Dataset/c_r2_101/c_r2_101/c_r2_101/c_r2_101.tar.gz', '/cache/c_r2_101.tar.gz')
# os.system('tar -xvf /cache/c_r2_101.tar.gz -C /cache')

hw_model_path = Context.get_model_path()
print(hw_model_path)

mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/524f086e8fd44fd98b09634d0228103b/Job/algo-mmdet2/mmdet2-4709/model/epoch_73.pth', '/cache/epoch_73.pth')
os.chdir('./mmdection')
os.system('pwd')
os.system('ls')
os.system('pip install pycocotools/')
os.system('python setup.py develop')


#os.system('python tools/test.py  custom_configs/eccv_coco_cascade_res2net_152_12x_m1.py  /cache/epoch_145.pth  --format-only --options "jsonfile_prefix=/cache/submission"')
#os.system('python tools/test.py  custom_configs/eccv_coco_cascade_senet154_12x_m2.py  /cache/epoch_145.pth  --format-only --options "jsonfile_prefix=/cache/submission"')
os.system('python tools/test.py  custom_configs/eccv_coco_detectors_resnest152_12x_m3.py  /cache/epoch_145.pth  --format-only --options "jsonfile_prefix=/cache/submission"')
mox.file.copy('/cache/submission.bbox.json', os.path.join(hw_model_path,  'm1.json'))