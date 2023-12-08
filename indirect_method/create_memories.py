import sys
import tqdm
import torch
from pathlib import Path
from dataset import indirectDataset
from network import torqueLstmNetwork, fsNetwork
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import init_weights, WINDOW, JOINTS, SKIP, max_torque, save, load_prev
from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = sys.argv[1]
train_path = join('..', 'bilateral_free_space_sep_27', 'train', 'psm1_mary', data)
root = Path('checkpoints' )
is_rnn = bool(int(sys.argv[2]))
if is_rnn:
    folder = 'lstm/' + data
    window = 1000
else:
    folder = 'ff/' + data
    window = WINDOW
in_joints = [0,1,2,3,4,5]
f = False

print('Running for is_rnn value: ', is_rnn)
                          
train_dataset = indirectDataset(train_path, window, SKIP, in_joints, is_rnn=is_rnn, filter_signal=f, return_prev_torque=True)
inputs = []
outputs = []


