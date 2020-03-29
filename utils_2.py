import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
Config['root_path'] = 'polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20
Config['batch_size'] = 64
# Config['debug'] = True
# Config['num_epochs'] = 2
# Config['batch_size'] = 128
Config['learning_rate'] = 0.001
Config['num_workers'] = 5