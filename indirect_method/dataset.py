import numpy as np
import os
from os.path import join
from scipy import signal
from torch.utils.data import Dataset

class indirectDataset(Dataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], num=1e9, is_rnn=False, filter_signal=False, return_prev_torque=False, train=True):

        all_joints = np.array([])
        joint_path = join(path, 'joints')
        cut_off = 1000

        all_joints = np.loadtxt(join(joint_path, 'interpolated_all_joints.csv'), delimiter=',')

        self.time = all_joints[:,0].astype('float64') # Don't know why the time get written out weird
        self.indices = np.array(indices)
        self.position = all_joints[:, self.indices + 1].astype('float32')
        self.velocity = all_joints[:, self.indices + 7].astype('float32')
        self.torque = all_joints[:, self.indices + 13].astype('float32')
        self.train = train

        if len(self.position.shape) < 2:
            self.position = np.expand_dims(self.position, axis=1)
            self.velocity = np.expand_dims(self.velocity, axis=1)
        self.window = window
        self.skip = skip
        self.is_rnn = is_rnn
        self.num = num
        self.return_prev_torque = return_prev_torque

        jacobian_path = join(path, 'jacobian')
        all_jacobian = np.loadtxt(join(jacobian_path, 'interpolated_all_jacobian.csv'), delimiter=',')
        self.jacobian = all_jacobian[:,1:].astype('float32')
        
        ## Filter signals
        if filter_signal:
            b, a = signal.butter(2, 0.02)
            for i in range(len(indices)):
#                self.position[:,i] = signal.filtfilt(b, a, self.position[:,i])
#                self.velocity[:,i] = signal.filtfilt(b, a, self.velocity[:,i])
                self.torque[:,i] = signal.filtfilt(b, a, self.torque[:,i])
        
    def __len__(self):
        return (int(min(self.torque.shape[0], self.num * 200)/self.window) - self.skip) if self.train else (self.torque.shape[0] - self.window)

    def __getitem__(self, idx):
        quotient = int(idx / self.skip)
        remainder = idx % self.skip
        begin = (quotient * self.window * self.skip + remainder) if self.train else idx
        end = begin + self.window * self.skip 
        if self.return_prev_torque: 
            begin = begin + 1 # New
            end = end + 1 # New
        return self.genitem(begin, end)

    def genitem(self, begin, end):
        position = self.position[begin:end:self.skip,:]
        velocity = self.velocity[begin:end:self.skip,:]
        if self.is_rnn:
            time = self.time[begin:end:self.skip].flatten()
            torque = self.torque[begin:end:self.skip,:]
            if self.return_prev_torque: 
                prev_torque = self.torque[begin-1:end-1:self.skip,:] # New
        else:
            time = self.time[end-self.skip]
            position = position.flatten()
            velocity = velocity.flatten()
            torque = self.torque[end-self.skip,:]
            if self.return_prev_torque: 
                prev_torque = self.torque[end-self.skip-1,:] # New

        jacobian = self.jacobian[begin:end:self.skip,:]

        if self.return_prev_torque: 
            return position, velocity, torque, jacobian, time, prev_torque # New
        else:
            return position, velocity, torque, jacobian, time


class indirectTestDataset(indirectDataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], is_rnn=False, filter_signal=False):
        super(indirectTestDataset, self).__init__(path, window, skip, indices = [0,1,2,3,4,5], is_rnn=is_rnn, filter_signal=filter_signal)

    def __len__(self):
        return self.torque.shape[0] - self.window*self.skip
        
    def __getitem__(self, idx):
        end = idx + self.window * self.skip 
        position, velocity, torque, time = self.genitem(idx, end)
        return position, velocity, torque, time
    
class indirectTrocarDataset(indirectDataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], num=1e9, seal='seal', filter_signal=False, net='ff'):
        super(indirectTrocarDataset, self).__init__(path, window, skip, indices = [0,1,2,3,4,5], is_rnn=False, filter_signal=filter_signal)
        self.fs_pred = np.loadtxt(path + '/' + net + '_' + seal + '_pred_filtered_torque.csv').astype('float32')
        self.fs_pred = self.fs_pred[:,1:]
        if net=='ff': #Throw out first window*skip if using ff
            self.time = self.time[window*skip:]
            self.position = self.position[window*skip:, :]
            self.velocity = self.velocity[window*skip:, :]
            self.torque = self.torque[window*skip:, :]
            
        self.num=num

    def __len__(self):
        return int(min(self.fs_pred.shape[0], self.num * 200)/self.window) - self.skip
        
    def __getitem__(self, idx):
        quotient = int(idx / self.skip)
        remainder = idx % self.skip
        begin = quotient * self.window * self.skip + remainder
        end = begin + self.window * self.skip
        position, velocity, torque, time = self.genitem(begin, end)
        fs_pred = self.fs_pred[end-self.skip,:]
#        fs_pred = fs_pred.flatten()
        return position, velocity, torque, time, fs_pred

class indirectTrocarTestDataset(indirectTrocarDataset):
    def __init__(self, path, window, skip, indices = [0,1,2,3,4,5], seal='base', filter_signal=False, net='ff'):
        super(indirectTrocarTestDataset, self).__init__(path, window, skip, indices = [0,1,2,3,4,5], seal=seal, filter_signal=filter_signal, net=net)
        
    def __len__(self):
        return self.fs_pred.shape[0] - self.window*self.skip -0 
        
    def __getitem__(self, idx):
        end = idx + self.window * self.skip
        position, velocity, torque, time = self.genitem(idx, end)
        jacobian = self.jacobian[end-self.skip, :]
        fs_pred = self.fs_pred[end-self.skip,:]
#        fs_pred = fs_pred.flatten()
        return position, velocity, torque, jacobian, time, fs_pred

