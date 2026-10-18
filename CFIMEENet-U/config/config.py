from easydict import EasyDict
import torch
import os
import logging

config = EasyDict()

config.means = (0.485, 0.456, 0.406)
config.stds = (0.229, 0.224, 0.225)

config.gpu = "1"

config.exp_name = "Uyghur"

config.num_workers = 24

config.batch_size = 12

config.max_epoch = 200
config.start_epoch = 0

config.lr = 1e-3
# using GPU
config.cuda = True

config.output_dir = 'output'

config.input_size = 640

config.max_annotation = 64

config.num_points = 20

config.use_hard = True

config.load_memory = False

config.scale = 1

config.grad_clip = 25

config.dis_threshold = 0.3

config.cls_threshold = 0.8

config.approx_factor = 0.004

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')

def logging_config(config):
    logging.info('==========Options============')
    for k, v in config.items():
        logging.info('{}: {}'.format(k, v))
    logging.info('=============End=============')
