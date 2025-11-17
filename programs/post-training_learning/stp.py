import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

dir = os.path.dirname(__file__)

class lif:
    def __init__(self, num_neuron=400, E_exc=0.0, E_inh=-100.0, E_rest=-65.0, V_reset=-65.0, V_thr=-52.0, tau=100.0, tau_ref=5.0, tau_e=1.0, tau_i=2.0, dt=0.1, offset=0.0, device='cuda'):
        try:
            self.device = torch.device(device)
        except Exception as e:
            print(f"Error initializing device: {e}")
        self.E_exc = E_exc
        self.E_inh = E_inh
        self.E_rest = E_rest
        self.num_neuron = num_neuron
        self.tau = tau
        self.tau_ref = tau_ref
        self.dt = dt
        self.V_thr_base = V_thr
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.offset = offset
        self.V_reset = V_reset
        self.V = torch.full((num_neuron,), E_rest - 40.0, device=self.device)
        self.last_spike_time = torch.full((num_neuron,), -float('inf'), device=self.device)
        self.g_e = torch.full((num_neuron,), 0.0, device=self.device)
        self.g_i = torch.full((num_neuron,), 0.0, device=self.device)
        self.V_thr = V_thr

    def reset(self):
        self.V = torch.full((self.num_neuron,), self.E_rest, device=self.device)
        self.last_spike_time = torch.full((self.num_neuron,), -float('inf'), device=self.device)

    def update_g_e(self, w_e):
        self.g_e -= self.g_e * (self.dt / self.tau_e)
        self.g_e += w_e

    def update_g_i(self, w_i):
        self.g_i -= self.g_i * (self.dt / self.tau_i)
        self.g_i += w_i


    def forward(self, w_e, w_i, t, theta=None):
        w_e = w_e.clamp(min=0.0)
        w_i = w_i.clamp(min=0.0)
        self.update_g_e(w_e)
        self.update_g_i(w_i)
        dV = (
            (self.E_exc - self.V) * self.g_e
            + (self.E_inh - self.V) * self.g_i
            + (self.E_rest - self.V)
        ) * (self.dt / self.tau)

        is_refractory = (t - self.last_spike_time) < self.tau_ref
        self.V[~is_refractory] += dV[~is_refractory]
        self.V[is_refractory] = self.V_reset

        if theta is not None:
            self.V_thr = theta + torch.full((self.num_neuron,), self.V_thr_base - self.offset, device=self.device)
        else:
            self.V_thr = torch.full((self.num_neuron,), self.V_thr_base - self.offset, device=self.device)

        spiked = (self.V >= self.V_thr) & (~is_refractory)
        self.V[spiked] = self.V_reset
        self.last_spike_time[spiked] = t

        return spiked
    
class STP:
    def __init__(self, omega_d=2.0, omega_f=3.33, U_0=0.6, w_ei=10.5, w_ie=17.0e-5, lr_pre=1e-4, lr_post=1e-2, tau_post1=15.0, tau_post2=30.0, tau_pre=20.0, k=10.5, input_size=28*28, num_neurons=400, dt=0.1, theta_plus=0.1, tau_theta=1e7, device='cuda'):
        try:
            self.device = torch.device(device)
        except Exception as e:
            print(f"Error initializing device: {e}")
        self.omega_d = omega_d
        self.omega_f = omega_f
        self.U_0 = U_0
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.lr_pre = lr_pre
        self.lr_post = lr_post
        self.tau_post1 = tau_post1
        self.tau_post2 = tau_post2
        self.tau_pre = tau_pre
        self.dt = dt
        self.k = k
        self.num_neurons = num_neurons
        self.theta_plus = theta_plus
        self.tau_theta = tau_theta
        self.input_size = input_size
        self.dtype = torch.float32
        self.theta = None
        self.load_theta()


        input_layer = lif(num_neuron=input_size, dt=dt, device=self.device)
        excitatory_layer = lif(num_neuron=self.num_neurons, dt=dt, offset=20.0, device=self.device)
        inhibitory_layer = lif(num_neuron=self.num_neurons, E_exc=0.0, E_inh=-85.0, E_rest=-60.0, V_reset=-50.0, V_thr=-40.0, tau=10.0, tau_ref=2.0, tau_e=1.0, tau_i=2.0, dt=dt, device=self.device)
        self.layers = {
            'input': input_layer,
            'excitatory': excitatory_layer,
            'inhibitory': inhibitory_layer
        }

        self.a_pre = torch.zeros_like(self.layers['input'].V, device=self.device)
        self.a_post1 = torch.zeros_like(self.layers['excitatory'].V, device=self.device)
        self.a_post2 = torch.zeros_like(self.layers['excitatory'].V, device=self.device)
        self.time = 0.0
        self.u_input = torch.full((input_size,), 0.0, device=self.device)
        self.x_input = torch.full((input_size,), 1.0, device=self.device)
        self.u_excitatory = torch.full((self.num_neurons,), 0.0, device=self.device)
        self.x_excitatory = torch.full((self.num_neurons,), 1.0, device=self.device)
        self.u_inhibitory = torch.full((self.num_neurons,), 0.0, device=self.device)
        self.x_inhibitory = torch.full((self.num_neurons,), 1.0, device=self.device)
        self.spike_input = torch.zeros((input_size,), dtype=torch.bool, device=self.device)
        self.spike_excitatory = torch.zeros((self.num_neurons,), dtype=torch.bool, device=self.device)
        self.spike_inhibitory = torch.zeros((self.num_neurons,), dtype=torch.bool, device=self.device)
        self.preferred_label = None
        self.classes = None
        self.max_rate = 2.0

        self.w_exc_inh = None
        self.w_inh_exc = None
        self.w_input_exc = None
        self.w_input_inh = None
        self.load_recurrent_weights()

    def load_theta(self, test_mode=False, weight_path='./random/'):
        if test_mode:
            theta_file = os.path.join(weight_path.replace('random', 'weights'), 'theta_A.npy')
        else:
            # 训练模式：初始化为默认值
            if self.theta is None:
                self.theta = torch.ones((self.num_neurons,), device=self.device) * 20.0
            return
        
        try:
            if os.path.exists(theta_file):
                # 加载Brian2格式的theta文件
                theta_data = np.load(theta_file)
                print(f"Loading theta from {theta_file}, shape: {theta_data.shape}")
                
                # Brian2的theta单位是volt，需要转换
                if theta_data.max() < 1.0:  # 如果最大值小于1，可能是以V为单位
                    theta_data = theta_data * 1000  # 转换为mV
                
                # 转换为张量
                theta_tensor = torch.from_numpy(theta_data.astype(np.float32)).to(self.device)
                
                # 确保尺寸匹配
                if theta_tensor.shape[0] != self.num_neurons:
                    print(f"Warning: theta size mismatch. Expected {self.num_neurons}, got {theta_tensor.shape[0]}")
                    if theta_tensor.shape[0] > self.num_neurons:
                        # 如果文件中的theta更多，截取前num_neurons个
                        theta_tensor = theta_tensor[:self.num_neurons]
                    else:
                        # 如果文件中的theta较少，扩展到匹配尺寸
                        extended_theta = torch.ones(self.num_neurons, device=self.device) * 20.0
                        extended_theta[:theta_tensor.shape[0]] = theta_tensor
                        theta_tensor = extended_theta
                
                self.theta = theta_tensor
                print(f"Loaded theta range: [{self.theta.min().item():.2f}, {self.theta.max().item():.2f}] mV")
                
            else:
                print(f"Theta file {theta_file} not found. Using default initialization.")
                self.theta = torch.ones((self.num_neurons,), device=self.device) * 20.0
                
        except Exception as e:
            print(f"Error loading theta: {e}")
            self.theta = torch.ones((self.num_neurons,), device=self.device) * 20.0

    def update_theta(self):
        self.theta -= self.theta * (self.dt /  self.tau_theta)
        self.theta[self.spike_excitatory] += self.theta_plus

    def load_recurrent_weights(self):
        aeai_file = os.path.join(dir + '/random/', 'AeAi.npy')
        if os.path.exists(aeai_file):
            self.w_exc_inh = self.load_brian2_weights(aeai_file)
        
        aiae_file = os.path.join(dir + '/random/', 'AiAe.npy')
        if os.path.exists(aiae_file):
            self.w_inh_exc = self.load_brian2_weights(aiae_file)
            
        xeae_file = os.path.join(dir + '/random/', 'XeAe.npy')
        if os.path.exists(xeae_file):
            self.w_input_exc = self.load_brian2_weights(xeae_file)
            
        xeai_file = os.path.join(dir + '/random/', 'XeAi.npy')
        if os.path.exists(xeai_file):
            self.w_input_inh = self.load_brian2_weights(xeai_file)
        
    def load_brian2_weights(self, weight_file):
        sparse_data = np.load(weight_file)
        print(f"Loading Brian2 weights from {weight_file}, shape: {sparse_data.shape}")
        
        # 根据文件名确定正确的权重矩阵尺寸
        filename = os.path.basename(weight_file)
        if 'XeAe' in filename:
            # 输入层到兴奋性层: (784, 400)
            weight_matrix = torch.zeros((self.input_size, self.num_neurons), 
                                       dtype=self.dtype, device=self.device)
            expected_i_max, expected_j_max = self.input_size - 1, self.num_neurons - 1
            
        elif 'XeAi' in filename:
            # 输入层到抑制性层: (784, 400)
            weight_matrix = torch.zeros((self.input_size, self.num_neurons), 
                                       dtype=self.dtype, device=self.device)
            expected_i_max, expected_j_max = self.input_size - 1, self.num_neurons - 1
            
        elif 'AeAi' in filename:
            # 兴奋性到抑制性: (400, 400)
            weight_matrix = torch.zeros((self.num_neurons, self.num_neurons), 
                                       dtype=self.dtype, device=self.device)
            expected_i_max, expected_j_max = self.num_neurons - 1, self.num_neurons - 1
            
        elif 'AiAe' in filename:
            # 抑制性到兴奋性: (400, 400)
            weight_matrix = torch.zeros((self.num_neurons, self.num_neurons), 
                                       dtype=self.dtype, device=self.device)
            expected_i_max, expected_j_max = self.num_neurons - 1, self.num_neurons - 1
            
        else:
            # 默认为输入到兴奋性连接
            print(f"Unknown weight file type: {filename}, using default (784, 400)")
            weight_matrix = torch.zeros((self.input_size, self.num_neurons), 
                                       dtype=self.dtype, device=self.device)
            expected_i_max, expected_j_max = self.input_size - 1, self.num_neurons - 1
        
        if sparse_data.size > 0 and len(sparse_data.shape) == 2:
            i_indices = sparse_data[:, 0].astype(int)
            j_indices = sparse_data[:, 1].astype(int)
            weights = sparse_data[:, 2].astype(np.float32)
            
            max_i = i_indices.max() if len(i_indices) > 0 else 0
            max_j = j_indices.max() if len(j_indices) > 0 else 0
            
            print(f"File: {filename}")
            print(f"Matrix size: {weight_matrix.shape}")
            print(f"Index ranges: i=[0,{max_i}], j=[0,{max_j}]")
            print(f"Expected: i=[0,{expected_i_max}], j=[0,{expected_j_max}]")
            
            # 检查索引是否在有效范围内
            valid_mask = (i_indices <= expected_i_max) & (j_indices <= expected_j_max)
            if not valid_mask.all():
                print(f"Warning: {(~valid_mask).sum()} invalid connections found")
                i_indices = i_indices[valid_mask]
                j_indices = j_indices[valid_mask]
                weights = weights[valid_mask]
            
            if len(i_indices) > 0:
                i_tensor = torch.from_numpy(i_indices).long()
                j_tensor = torch.from_numpy(j_indices).long()
                w_tensor = torch.from_numpy(weights).to(device=self.device, dtype=self.dtype)
                
                weight_matrix[i_tensor, j_tensor] = w_tensor
                
                print(f"Successfully loaded {len(weights)} connections")
                print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
                print(f"Non-zero weights: {(weight_matrix > 0).sum().item()}")
            else:
                print("No valid connections found")
                
        return weight_matrix

    def reset(self):
        for layer in self.layers.values():
            layer.reset()
        self.time = 0.0
        self.u_input = torch.zeros_like(self.layers['input'].V, device=self.device)
        self.x_input = torch.ones_like(self.layers['input'].V, device=self.device)
        self.u_excitatory = torch.zeros_like(self.layers['excitatory'].V, device=self.device)
        self.x_excitatory = torch.ones_like(self.layers['excitatory'].V, device=self.device)
        self.u_inhibitory = torch.zeros_like(self.layers['inhibitory'].V, device=self.device)
        self.x_inhibitory = torch.ones_like(self.layers['inhibitory'].V, device=self.device)
        self.spike_input = torch.zeros_like(self.layers['input'].V, dtype=torch.bool, device=self.device)
        self.spike_excitatory = torch.zeros_like(self.layers['excitatory'].V, dtype=torch.bool, device=self.device)
        self.spike_inhibitory = torch.zeros_like(self.layers['inhibitory'].V, dtype=torch.bool, device=self.device)

    def STDP(self):
        self.a_pre -= self.a_pre * self.dt / self.tau_pre
        self.a_post1 -= self.a_post1 * self.dt / self.tau_post1
        self.a_post2 -= self.a_post2 * self.dt / self.tau_post2

        if self.spike_input.any():
            dw = self.lr_pre * self.a_post1.unsqueeze(0)
            self.w_input_exc[self.spike_input, :] -= dw
            self.a_pre[self.spike_input] = 1.0

        if self.spike_excitatory.any():
            dw = self.lr_post * (self.a_pre.unsqueeze(1) * self.a_post2.unsqueeze(0))
            self.w_input_exc[:, self.spike_excitatory] += dw[:, self.spike_excitatory]
            self.a_post1[self.spike_excitatory] = 1.0
            self.a_post2[self.spike_excitatory] = 1.0

        self.w_input_exc = self.w_input_exc.clamp(min=0.0)

    def STP(self):
        self.u_input -= self.u_input * (self.dt * self.omega_f)
        self.x_input += self.omega_d * (1 - self.x_input) * self.dt

        self.u_input[self.spike_input] = self.u_input[self.spike_input] + self.U_0 * (1 - self.u_input[self.spike_input])
        r_input = self.u_input * self.x_input
        self.x_input[self.spike_input] = self.x_input[self.spike_input] - r_input[self.spike_input]

        return r_input
    
    def poisson_spike_train(self, img, simtime=350.0, dt=1.0, max_rate=2.0):
        img_flat = img.flatten()
        
        T = int(simtime / dt)

        rates = img_flat / 8.0 * max_rate
        p_spike = rates.float() * (dt / 1000.0)
        lam = p_spike.unsqueeze(0).repeat(T, 1)

        counts = torch.poisson(lam)
        spikes = (counts > 0).float()

        return spikes
    
    def normalize_weights(self):
        row_sums = self.w_input_exc.sum(dim=0, keepdim=True)
        self.w_input_exc = self.w_input_exc / row_sums * 78.0

    def active_inhibition(self, excitatory_input, recent_activity, inhibition_strength=0.5):
        inhibition = recent_activity.float().mean() * inhibition_strength
        output = excitatory_input - inhibition
        return output
    
    def recent_excitatory_activity_update(self, recent_excitatory_activity):
        recent_excitatory_activity -= recent_excitatory_activity * self.dt / 200.0
        recent_excitatory_activity[self.spike_excitatory] += 1.0
        return recent_excitatory_activity
    
    def run(self, input_data=None, simtime=350.0, train=True, STP_on=True):
        if train:
            self.normalize_weights()

        input_data = input_data.reshape(-1) if input_data is not None else None
        input_data = input_data.to(self.device) if input_data is not None else None
        if input_data is None:
            input_data = torch.zeros((self.layers['input'].num_neuron,), dtype=torch.bool, device=self.device)

        spikes = self.poisson_spike_train(input_data, simtime=simtime, dt=self.dt, max_rate=self.max_rate)

        num_steps = int(simtime / self.dt)
        spike_record_excitatory = torch.zeros((self.layers['excitatory'].num_neuron, num_steps), dtype=torch.bool, device=self.device)
        spike_record_input = torch.zeros((self.layers['input'].num_neuron, num_steps), dtype=torch.bool, device=self.device)
        recent_excitatory_activity = torch.zeros((self.layers['excitatory'].num_neuron,), device=self.device)

        for step in range(num_steps):
            #self.spike_input = self.layers['input'].forward(w_e=self.w_ei * 1e3 * spikes[step].float(), 
            #                                                w_i=torch.zeros((self.layers['input'].num_neuron,), dtype=torch.float32, device=self.device),
            #                                                t=self.time)
            self.spike_input = spikes[step].bool()
            if STP_on:
                r_input = self.STP()
                w_effective = self.k * torch.mv(self.w_input_exc.T, r_input.float())
            else:
                w_effective = torch.zeros_like(self.w_input_exc, device=self.device)

            excitatory_input = self.active_inhibition(torch.mv(self.w_input_exc.T, self.spike_input.float()) + w_effective, recent_excitatory_activity, inhibition_strength=0.5)

            self.spike_excitatory = self.layers['excitatory'].forward(w_e=excitatory_input, 
                                                                    w_i=self.w_ie * (torch.mv(self.w_inh_exc.T, self.spike_inhibitory.float())),
                                                                    t=self.time,
                                                                    theta=self.theta)
            self.spike_inhibitory = self.layers['inhibitory'].forward(w_e=self.w_ei * (torch.mv(self.w_exc_inh.T, self.spike_excitatory.float())), 
                                                                    w_i=torch.zeros((400,), dtype=torch.float64, device=self.device), 
                                                                    t=self.time)
            if train:
                self.update_theta()
                self.STDP()
            spike_record_input[:, step] = self.spike_input
            spike_record_excitatory[:, step] = self.spike_excitatory
            self.time += self.dt
            recent_excitatory_activity = self.recent_excitatory_activity_update(recent_excitatory_activity)
            print(f'V: {self.layers["excitatory"].V.mean():>10.5f} | V_thr mean: {self.layers["excitatory"].V_thr.mean():.2f} | {self.layers["inhibitory"].V.mean():>10.5f} | {self.layers["inhibitory"].V_thr.mean():.2f} | inh_g_e: {self.layers["inhibitory"].g_e.mean():>10.5f} | {self.spike_input.sum():>10} | {self.spike_excitatory.sum():>10} | {self.spike_inhibitory.sum():>10} | {self.theta.mean():.2f} | {self.layers["excitatory"].g_e.sum():>10.5f} | g_i: {self.layers["excitatory"].g_i.sum():>10.5f} | time: {self.time:.1f} ms', end='\r')
        
        '''
        spike_record_excitatory_mask = spike_record_excitatory.float().sum(dim=1) > 0
        spike_record_input_mask = spike_record_input.float().sum(dim=1) > 0
        spike_record_excitatory_cont = (spike_record_excitatory_mask).sum().item()
        if train:    
            if spike_record_excitatory_cont < 20:
                self.w[spike_record_input_mask, :] += 0.2 * (20.0 - float(spike_record_excitatory_cont)) / 10.0
                self.w[spike_record_input_mask, :] += (self.w[spike_record_input_mask, :].mean() - self.w[spike_record_input_mask, :]) * 1.01
                self.w = self.w.clamp(min=0.0, max=1.0)
            if spike_record_excitatory_cont > 100:
                self.w[spike_record_input_mask, :] -= 0.1 * (float(spike_record_excitatory_cont) - 100.0) / 50.0
                self.w[spike_record_input_mask, :] += (self.w[spike_record_input_mask, :].mean() - self.w[spike_record_input_mask, :]) * 0.05
                self.w = self.w.clamp(min=0.0, max=1.0)
        '''

        return spike_record_input, spike_record_excitatory

    def label_neurons(self, train_data, update_interval=200, STP_on=True):
        num_neurons = self.layers['excitatory'].num_neuron
        
        result_monitor = torch.zeros((update_interval, num_neurons), device=self.device)
        input_numbers = []
        
        assignments = torch.zeros(num_neurons, device=self.device)
        
        for j, (img, label) in enumerate(train_data):
            while True:
                _, spike_record = self.run(input_data=img, simtime=350.0, train=False, STP_on=STP_on)
                self.run(train=False, simtime=150.0, STP_on=STP_on)
                if spike_record.float().sum(dim=0).max().item() >= 5:
                    self.reset_max_rate()
                    break
                else:
                    self.plus_max_rate(increment=1.0)
            
            current_spike_count = spike_record.sum(dim=1).float()
            result_monitor[j % update_interval] = current_spike_count
            input_numbers.append(label)
            
            if j % update_interval == update_interval - 1 and j > 0:
                print(f"Updating assignments at sample {j+1}")
                
                current_labels = input_numbers[-update_interval:]
                
                assignments = self.get_new_assignments_torch(
                    result_monitor, torch.tensor(current_labels, device=self.device)
                )
                
                print(f"Assignment update: {torch.bincount(assignments.long(), minlength=10)}")
        
        if len(input_numbers) % update_interval != 0:
            remaining_samples = len(input_numbers) % update_interval
            current_labels = input_numbers[-remaining_samples:]
            assignments = self.get_new_assignments_torch(
                result_monitor[:remaining_samples], 
                torch.tensor(current_labels, device=self.device)
            )
        
        self.preferred_label = assignments.long()
        self.classes = torch.unique(self.preferred_label).sort().values
        return self.preferred_label

    def get_new_assignments_torch(self, result_monitor, input_numbers):
        n_e = self.layers['excitatory'].num_neuron
        assignments = torch.zeros(n_e, device=self.device)
        maximum_rate = torch.zeros(n_e, device=self.device)
        
        for j in range(10):
            class_mask = (input_numbers == j)
            num_assignments = class_mask.sum().item()
            
            if num_assignments > 0:
                class_responses = result_monitor[class_mask]
                rate = class_responses.mean(dim=0)
                
                better_response = rate > maximum_rate
                maximum_rate[better_response] = rate[better_response]
                assignments[better_response] = j
        
        return assignments
    
    def get_label(self, input_data, STP_on=True):
        while True:
            _, spike_record = self.run(input_data=input_data, simtime=350.0, train=False, STP_on=STP_on)
            self.run(train=False, simtime=150.0, STP_on=STP_on)
            if spike_record.float().sum(dim=0).max().item() >= 5:
                self.reset_max_rate()
                print(f"response good. Reset max rate to {self.max_rate}.")
                break
            else:
                self.plus_max_rate(increment=1.0)
                print(f"Increased max rate to {self.max_rate} to improve response.")

        spike_count = spike_record.sum(dim=1)
        if self.preferred_label is None:
            raise ValueError("Neurons have not been labeled yet. Please run label_neurons() first.")
        if spike_count.sum() == 0:
            return None
        
        votes = torch.zeros((len(self.classes),), device=self.device)
        for i, label in enumerate(self.classes):
            votes[i] = spike_count[self.preferred_label == label].sum()

        best_class_idx = torch.argmax(votes)
        if votes[best_class_idx] == 0:
            return None

        return self.classes[best_class_idx].item()
    
    def reset_max_rate(self):
        self.max_rate = 2.0

    def plus_max_rate(self, increment=0.5):
        self.max_rate += increment


def test():
    model = STP(dt=0.1)

    train_data = []
    for label in range(10):
        for _ in range(5):
            img = torch.zeros(28*28, dtype=torch.bool)
            start = label * 70
            end = start + 70
            img[start:end] = (torch.rand(end - start) > 0.5).to(torch.float32)
            train_data.append((img, label))

    print("Training the network...")
    for img, label in train_data:
        model.run(input_data=img, simtime=350.0, train=True)
        model.run(train=False, simtime=150.0)

    print("Training completed!")

    print("Labeling neurons based on their responses...")
    preferred_label = model.label_neurons(train_data, update_interval=200)
    print("Labeling completed!")
    
    print("Preferred labels of neurons (first 30 neurons):")
    print(preferred_label[:30])
    label_count = pd.Series(preferred_label.numpy()).value_counts().sort_index()
    print("\nDistribution of neurons per class:")
    print(label_count)

def get_mnist_data(batch_size=1, n_per_class=10, train=True):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.PILToTensor()
    ])
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    targets = np.array(dataset.targets)
    indices = []
    for i in range(10):
        class_indices = np.where(targets == i)[0]
        selected_indices = np.random.choice(class_indices, size=n_per_class, replace=False)
        indices.extend(selected_indices)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader

def mnist_test():
    model = STP(dt=0.1)
    if os.path.exists("saved_models/weights.pt"):
        w = torch.load("saved_models/weights.pt", map_location=model.device)
        theta = torch.load("saved_models/theta.pt", map_location=model.device)
        model.w_input_exc = w
        model.theta = theta
        gain = torch.load("saved_models/best_gain.pt")
        print("Loaded pre-trained weights.")
    else:
        print("No pre-trained weights found.")
        gain = torch.tensor([])

    for _ in range(3):
        train_loader = get_mnist_data(batch_size=1, n_per_class=100, train=True)

        print("Training the network...")
        for (img, label) in list(train_loader)[:800]:
            img, label = img.squeeze(0), label.item()
            print(f"Training sample label: {label}")
            while True:
                input_spikking_record, spike_record = model.run(input_data=img, simtime=350.0, train=True, STP_on=True)
                model.run(train=False, simtime=150.0, STP_on=True)
                spike_count = spike_record.float().sum(dim=1).cpu().numpy()
                top10_idx = np.argsort(spike_count)[-10:][::-1]
                print(f'{(spike_record.float().sum(dim=1) > 0).sum().item()} neurons responded. {spike_record.float().sum(dim=0).max().item()} max response count in a single instance. Neurons {top10_idx} response counts')
                if spike_record.float().sum(dim=0).max().item() >= 5:
                    model.reset_max_rate()
                    print(f"Sample {label} responded well. Reset max rate to {model.max_rate}.")
                    break
                else:
                    model.plus_max_rate(increment=1.0)
                    print(f"Increased max rate to {model.max_rate} to improve response.")

        print("Training completed!")

        print("Labeling neurons based on their responses...")
        mnist_samples = [(img.squeeze(0), label.item()) for img, label in list(train_loader)[:800]]
        preferred_label = model.label_neurons(mnist_samples, update_interval=200, STP_on=True)
        print("Labeling completed!")

        print("Preferred labels of neurons (first 30 neurons):")
        print(preferred_label[:30])

        label_count = pd.Series(preferred_label.cpu().numpy()).value_counts().sort_index()
        print("\nDistribution of neurons per class:")
        for i, count in label_count.items():
            print(f"{i}: {count}")

        print("\nTesting phase (prediction accuracy):")
        pred_matrix = torch.zeros((10, 10), dtype=torch.int32)
        correct, total = 0, 0
        for (img, label) in list(train_loader)[800:]:
            img = img.squeeze(0)
            pred = model.get_label(img, STP_on=True)
            if pred is not None and pred == label.item():
                correct += 1
            total += 1
            if pred is not None:
                pred_matrix[label.item(), pred] += 1

        if os.path.exists(dir + "/saved_models") == False:
            os.makedirs(dir + "/saved_models", exist_ok=True)
        timestemp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        gain = torch.cat((gain, torch.tensor([100 * correct / total])))
        torch.save(model.w_input_exc, dir + "/saved_models/weights.pt")
        torch.save(model.theta, dir + "/saved_models/theta.pt")
        torch.save(gain, dir + "/saved_models/best_gain.pt")
        torch.save(input_spikking_record, dir + f"/saved_models/input_spikking_history{timestemp}.pt")
        torch.save(spike_record, dir + f"/saved_models/exc_spiking_history{timestemp}.pt")
        torch.save(pred_matrix, dir + f"/saved_models/pred_matrix{timestemp}.pt")
        print(f"Prediction accuracy: {correct}/{total} = {100 * correct / total:.2f}%")


if __name__ == "__main__":
    mnist_test()
