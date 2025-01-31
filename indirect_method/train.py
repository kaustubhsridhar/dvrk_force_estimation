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
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = sys.argv[1]
root = Path('checkpoints' )
is_rnn = bool(int(sys.argv[2]))
arm_name = sys.argv[3]
train_path = join('..', 'bilateral_free_space_sep_27', 'train', arm_name, data)
val_path = join('..', 'bilateral_free_space_sep_27', 'val', arm_name, data)
if is_rnn:
    folder = 'lstm/' + data
else:
    folder = 'ff/' + data
range_torque = torch.tensor(max_torque).to(device)
    
lr = 1e-3
batch_size = 128
epochs = 1000
validate_each = 5
use_previous_model = False
epoch_to_use = 40
in_joints = [0,1,2,3,4,5]
f = False
print('Running for is_rnn value: ', is_rnn)
os.makedirs(root, exist_ok=True)
os.makedirs(root / f'filtered_torque', exist_ok=True)
os.makedirs(root / f'filtered_torque' / arm_name, exist_ok=True)
os.makedirs(root / "filtered_torque" / arm_name / folder, exist_ok=True)


networks = []
optimizers = []
schedulers = []
model_root = []

for j in range(JOINTS):
    if is_rnn:
        window = 1000
        networks.append(torqueLstmNetwork(batch_size, device))
    else:
        window = WINDOW
        networks.append(fsNetwork(window))

    networks[j].to(device)
    optimizers.append(torch.optim.Adam(networks[j].parameters(), lr))
    schedulers.append(ReduceLROnPlateau(optimizers[j], verbose=True))
                          
train_dataset = indirectDataset(train_path, window, SKIP, in_joints, is_rnn=is_rnn, filter_signal=f)
val_dataset = indirectDataset(val_path, window, SKIP, in_joints, is_rnn=is_rnn, filter_signal=f)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=False)
    
loss_fn = torch.nn.MSELoss()

for j in range(JOINTS):
    try:
        model_root.append(root / "filtered_torque" / arm_name / (folder + str(j)))
        model_root[j].mkdir(mode=0o777, parents=False)
    except OSError:
        print("Model path exists")

if use_previous_model:
    for j in range(JOINTS):
        epoch = load_prev(networks[j], model_root[j], epoch_to_use, optimizers[j], schedulers[j])
else:
    for j in range(JOINTS):
        init_weights(networks[j])
    epoch = 1

print('Training for ' + str(epochs))
best_loss = torch.zeros(6) + 1e8

for e in range(epoch, epochs + 1):

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(e, optimizers[0].param_groups[0]['lr']))
    epoch_loss = 0

    for j in range(JOINTS):
        networks[j].train()
    
    for i, (position, velocity, torque, jacobian, time) in enumerate(train_loader):
        position = position.to(device)
        velocity = velocity.to(device)
        torque = torque.to(device)
        if is_rnn: 
            posvel = torch.cat((position, velocity), axis=2).contiguous()
        else:
            posvel = torch.cat((position, velocity), axis=1).contiguous()

        step_loss = 0

        for j in range(JOINTS):
            pred = networks[j](posvel) * range_torque[j]
            if is_rnn:
                loss = loss_fn(pred.squeeze(), torque[:,:,j])
            else:
                loss = loss_fn(pred.squeeze(), torque[:,j])
            step_loss += loss.item()
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()

        tq.update(batch_size)
        tq.set_postfix(loss=' loss={:.5f}'.format(step_loss))
        epoch_loss += step_loss

    tq.set_postfix(loss=' epoch loss={:.5f}'.format(epoch_loss/len(train_loader)))
    
    if e % validate_each == 0:
        for j in range(JOINTS):
            networks[j].eval()

        val_loss = torch.zeros(JOINTS)
        for i, (position, velocity, torque, jacobian, time) in enumerate(val_loader):
            position = position.to(device)
            velocity = velocity.to(device)
            torque = torque.to(device)
            if is_rnn: 
                posvel = torch.cat((position, velocity), axis=2).contiguous()
            else:
                posvel = torch.cat((position, velocity), axis=1).contiguous()

            for j in range(JOINTS):
                pred = networks[j](posvel) * range_torque[j]
                if is_rnn:
                    loss = loss_fn(pred.squeeze(), torque[:,:,j])
                else:
                    loss = loss_fn(pred.squeeze(), torque[:,j])
                val_loss[j] += loss.item()

        val_loss = val_loss / len(val_loader) # New: moved out of for loop!
                
        for j in range(JOINTS):
            schedulers[j].step(val_loss[j])
        tq.set_postfix(loss='validation loss={:5f}'.format(torch.sum(val_loss))) # New: replaced torch.mean with torch.sum

        for j in range(JOINTS):
            model_path = model_root[j] / "model_joint_{}.pt".format(e)
            save(e, networks[j], model_path, val_loss[j], optimizers[j], schedulers[j])

            if val_loss[j] < best_loss[j]:
                model_path = model_root[j] / "model_joint_best.pt"
                save(e, networks[j], model_path, val_loss[j], optimizers[j], schedulers[j])
                best_loss[j] = val_loss[j]

        
    tq.close()
