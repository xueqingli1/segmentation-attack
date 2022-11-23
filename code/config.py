# config.py
import os
CLASS_NUM = 4
DIM = 2
LOSS_MODE = 'multiclass'
DEBUG = True

RESULT_DIR = 'results'
os.makedirs(RESULT_DIR, exist_ok=True)


MODEL = 'unet'
ENCODER = 'resnet34'
LOSS_TYPE = 'focal'
ENCODER_WEIGHT = 'imagenet'
GPU_ID = 4
# MODEL = 'fpn'
# encoder = 'resnet34'
# encoder_weights = 'imagenet'
