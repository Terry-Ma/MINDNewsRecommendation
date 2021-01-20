import logging
import yaml
import os
import torch

from gensim.models import KeyedVectors
from torch import nn

logger = logging.getLogger()

def set_logger(log_path):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

def checkpoint_process(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        logger.info('checkpoint dir not exist, make dir {}'.format(checkpoint_dir))
    else:
        logger.info('checkpoint dir exist: {}'.format(checkpoint_dir))

def generate_config(args):
    with open('./config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info('load config from {}'.format('./config.yml'))
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            find_k = False
            for _, config_v in config.items():
                if key in config_v:
                    config_v[key] = value
                    find_k = True
                    break
            if not find_k:
                config['train'][key] = value
    config['train']['checkpoint_path'] = './checkpoint/{}/'.format(args.experiment_name)
    config['train']['submit_path'] = '../submit/{}.csv'.format(args.experiment_name)
    config['train']['tb_path'] = './train_log/tensorboard/{}/'.format(args.experiment_name)
    logger.info('will use config \n{}'.format(config))

    return config

if __name__ == '__main__':
    set_logger('./NRMS/train_log/train.log')
    logger.info('set logger sucess')