""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#directory to save weights file
CHECKPOINT_PATH = 'log'

#total training epoches
EPOCH = 120
MILESTONES = [25, 60, 80]

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
