import sys
import torch
from network import *
import torch.nn as nn
import utils
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import indirectTestDataset, indirectDataset
from os.path import join 
import matplotlib.pyplot as plt
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

JOINTS = utils.JOINTS
epoch_to_use = 1000 #int(sys.argv[1])
exp = sys.argv[1] 
net = sys.argv[2]
data = sys.argv[3]
preprocess = 'filtered_torque'# sys.argv[4]
is_rnn = ('lstm' in net)
print('Running for is_rnn value: ', is_rnn)
if is_rnn:
    batch_size = 1
else:
    batch_size = 8192
root = Path('checkpoints')
    
def main():
    all_pred = None
    path = join('..', 'bilateral_free_space_sep_27', exp, 'psm1_mary', data)
    in_joints = [0,1,2,3,4,5]

    if is_rnn:
        window = 1000
    else:
        window = utils.WINDOW
    
    if is_rnn:
        dataset = indirectDataset(path, window, utils.SKIP, in_joints, is_rnn=is_rnn, return_prev_torque=True)
    else:
        dataset = indirectTestDataset(path, window, utils.SKIP, in_joints, is_rnn=is_rnn, return_prev_torque=True)
    loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=False, drop_last=False)

    model_root = []    
    for j in range(JOINTS):
        folder = data + str(j)        
        model_root.append(root / preprocess / net / folder)
        
    networks = []
    for j in range(JOINTS):
        if is_rnn:
            networks.append(torqueLstmNetwork(batch_size, device).to(device))
        else:
            networks.append(fsNetwork(window).to(device))

    for j in range(JOINTS):
        utils.load_prev(networks[j], model_root[j], epoch_to_use)
        print("Loaded a " + str(j) + " model")

    loss_fn = torch.nn.MSELoss()

    for i, (global_position, global_velocity, global_torque, global_jacobian, global_time, global_prev_torque) in enumerate(loader):
        # print(global_position.shape, global_velocity.shape, global_torque.shape, global_jacobian.shape, global_time.shape, global_prev_torque.shape) 
        # torch.Size([1, 1000, 6]) torch.Size([1, 1000, 6]) torch.Size([1, 1000, 6]) torch.Size([1, 1000, 36]) torch.Size([1, 1000]) torch.Size([1, 1000, 6])

        if i == 0:
            for i2 in range(1, global_position.shape[1]+1): # have to iterate over every timestep till 1000 for the first window of size 1000!
                position = global_position[:,:i2,:].to(device)
                velocity = global_velocity[:,:i2,:].to(device)
                torque = global_torque[:,:i2,:]
                time = global_time[:,:i2].to(device)
                if i2 == 1:
                    last_trues = global_prev_torque[0,:1,:].cpu().numpy().tolist() # (1, 6)
                    last_preds = global_prev_torque[0,:1,:].cpu().numpy().tolist() # (1, 6)
                    times = [global_time[0,1].cpu().item()] # (1,)
                prev_torque = global_prev_torque[:,:i2,:] # (1, i2, 6)

                position = position.to(device)
                velocity = velocity.to(device)
                posvel = torch.cat((position, velocity), axis=2 if is_rnn else 1).contiguous()

                cur_pred = torch.zeros(torque.shape) # (1, i2, 6)
                for j in range(JOINTS):
                    pred = networks[j](posvel).detach().cpu() # (1, i2, 1)
                    cur_pred[:, :, j] = pred[:, :, 0] + prev_torque[:, :, j] # (1, i2)

                last_trues.append(torque[0,-1,:].cpu().numpy().tolist())
                last_preds.append(cur_pred[0,-1,:].cpu().numpy().tolist()) # (i2+1, 6)
                times.append(time[0,-1].cpu().item())
                print(f'At {i}/{len(loader)}; Processing {i2}/{global_position.shape[1]}, MSE So Far: {loss_fn(torch.tensor(last_preds), torch.tensor(last_trues)).item()}')
        else:
            position = global_position
            velocity = global_velocity
            torque = global_torque
            prev_torque = global_prev_torque
            time = global_time

            position = position.to(device)
            velocity = velocity.to(device)
            posvel = torch.cat((position, velocity), axis=2 if is_rnn else 1).contiguous()

            cur_pred = torch.zeros(torque.shape) # (1, 1000, 6)
            for j in range(JOINTS):
                pred = networks[j](posvel).detach().cpu() # (1, 1000, 1)
                cur_pred[:, :, j] = pred[:, :, 0] + prev_torque[:, :, j] # (1, 1000)

            last_trues.append(torque[0,-1,:].cpu().numpy().tolist())
            last_preds.append(cur_pred[0,-1,:].cpu().numpy().tolist())
            times.append(time[0,-1].cpu().item())
            print(f'At {i}/{len(loader)}; Current window MSE: {loss_fn(torque, cur_pred).item()}, MSE So Far: {loss_fn(torch.tensor(last_preds), torch.tensor(last_trues)).item()}')

    last_trues = np.array(last_trues) 
    last_preds = np.array(last_preds)
    print(f'Last trues: {last_trues.shape}, last preds: {last_preds.shape}')
    print(f'Scaled MSE: {loss_fn(torch.tensor(last_preds), torch.tensor(last_trues)).item()}')

    # last_trues.shape = (1000, 6)
    # last_preds.shape = (1000, 6)
    # plot 6 plots one below the other comparing the true and predicted torque for each of the 6 joints
    plt.figure(figsize=(6, 12))
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.plot(times, last_trues[:,i], label='True')
        plt.plot(times, last_preds[:,i], label='Predicted')
        plt.legend()
    os.makedirs('../images', exist_ok=True)
    plt.savefig(f'../images/bilateral_free_space_sep_27___{exp}___psm1_mary___{data}___last_trues_preds.png')


if __name__ == "__main__":
    main()
