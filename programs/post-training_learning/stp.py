import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class lif:
    def __init__(self, num_neuron=400, E_exc=0.0, E_inh=-100.0, E_rest=-65.0, V_thr=-52.0, tau=100.0, tau_ref=2.0, tau_e=2.0, tau_i=1.0, dt=0.1, device='cuda'):
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
        self.V_thr = V_thr
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.V = torch.full((num_neuron,), E_rest, device=self.device)
        self.last_spike_time = torch.full((num_neuron,), -float('inf'), device=self.device)
        self.g_e = torch.full((num_neuron,), 0.0, device=self.device)
        self.g_i = torch.full((num_neuron,), 0.0, device=self.device)

    def reset(self):
        self.V = torch.full((self.num_neuron,), self.E_rest, device=self.device)
        self.last_spike_time = torch.full((self.num_neuron,), -float('inf'), device=self.device)

    def update_g(self, w_e, w_i, is_feedback=False):
        if not is_feedback: 
            self.g_e -= self.g_e * (self.dt / self.tau_e)
            self.g_i -= self.g_i * (self.dt / self.tau_i)

        self.g_e += w_e
        self.g_i += w_i

    def forward(self, w_e, w_i, t):
        self.update_g(w_e, w_i)
        dV = (
            (self.E_exc - self.V) * self.g_e
            + (self.E_inh - self.V) * self.g_i
            + (self.E_rest - self.V)
        ) * (self.dt / self.tau)
        self.V += dV

        is_refractory = (t - self.last_spike_time) < self.tau_ref
        self.V[is_refractory] = self.E_rest

        spiked = self.V >= self.V_thr
        self.V[spiked] = self.E_rest
        self.last_spike_time[spiked] = t

        return spiked
    
class STP:
    def __init__(self, omega_d=2.0, omega_f=3.33, U_0=0.6, w_ei=10.5, w_ie=17.0, lr_pre=1e-4, lr_post=1e-2, tau_post1=1.0, tau_post2=2.0, tau_pre=1.5, k=0.6, input_size=28*28, num_neurons=400, dt=0.1, device='cuda'):
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

        input_layer = lif(num_neuron=input_size, dt=dt, device=self.device)
        excitatory_layer = lif(num_neuron=self.num_neurons, dt=dt, device=self.device)
        inhibitory_layer = lif(num_neuron=self.num_neurons, E_exc=0.0, E_inh=-85.0, E_rest=-60.0, V_thr=-40.0, tau=10.0, tau_ref=5.0, tau_e=2.0, tau_i=1.0, dt=dt, device=self.device)
        self.layers = {
            'input': input_layer,
            'excitatory': excitatory_layer,
            'inhibitory': inhibitory_layer
        }

        self.a_pre = torch.zeros_like(self.layers['input'].V, device=self.device)
        self.a_post1 = torch.zeros_like(self.layers['excitatory'].V, device=self.device)
        self.a_post2 = torch.zeros_like(self.layers['excitatory'].V, device=self.device)
        self.time = 0.0
        self.w = torch.randn((input_size, self.num_neurons), device=self.device).abs()
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
            self.a_pre[self.spike_input] = 1.0
            dw = self.lr_pre * self.a_post1.unsqueeze(0)
            self.w[self.spike_input, :] -= dw

        if self.spike_excitatory.any():
            self.a_post1[self.spike_excitatory] = 1.0
            self.a_post2[self.spike_excitatory] = 1.0
            dw = self.lr_post * (self.a_pre.unsqueeze(1) * self.a_post2.unsqueeze(0))
            self.w[:, self.spike_excitatory] += dw[:, self.spike_excitatory]

    def STP(self):
        self.u_input -= self.u_input * (self.dt * self.omega_f)
        self.x_input += self.omega_d * (1 - self.x_input) * self.dt

        self.u_input[self.spike_input] = self.u_input[self.spike_input] + self.U_0 * (1 - self.u_input[self.spike_input])
        r_input = self.u_input * self.x_input
        self.x_input[self.spike_input] = self.x_input[self.spike_input] - r_input[self.spike_input]

        return r_input
    
    def run(self, input_data=None, simtime=350.0, train=True, STP_on=True):
        input_data = input_data.reshape(-1) if input_data is not None else None
        input_data = input_data.to(self.device) if input_data is not None else None
        if input_data is None:
            input_data = torch.zeros((self.layers['input'].num_neuron,), dtype=torch.bool, device=self.device)
        input_data = input_data

        num_steps = int(simtime / self.dt)
        spike_record_excitatory = torch.zeros((self.layers['excitatory'].num_neuron, num_steps), dtype=torch.bool)
        spike_record_input = torch.zeros((self.layers['input'].num_neuron, num_steps), dtype=torch.bool)
        for step in range(num_steps):
            self.spike_input = self.layers['input'].forward(w_e=input_data, 
                                                            w_i=torch.zeros((self.layers['input'].num_neuron,), dtype=torch.float32, device=self.device),
                                                            t=self.time)
            if STP_on:
                r_input = self.STP()
                w_effective = self.k * self.w * r_input.unsqueeze(1)
            else:
                w_effective = torch.zeros_like(self.w, device=self.device)
            self.spike_excitatory = self.layers['excitatory'].forward((self.w + w_effective)[self.spike_input].sum(dim=0), 
                                                                      torch.zeros((400,), dtype=torch.float32, device=self.device),
                                                                      self.time)
            self.spike_inhibitory = self.layers['inhibitory'].forward(self.w_ei * self.spike_excitatory.float(), 
                                                                      torch.zeros((400,), dtype=torch.float32, device=self.device), 
                                                                      self.time)
            self.layers['excitatory'].update_g(w_e=torch.zeros((400,), dtype=torch.float32, device=self.device),
                                               w_i=self.w_ie * (torch.full_like(self.spike_inhibitory.float(), self.spike_inhibitory.sum(), device=self.device) - self.spike_inhibitory.float()), 
                                               is_feedback=True)
            if train:
                self.STDP()
            spike_record_input[:, step] = self.spike_input
            spike_record_excitatory[:, step] = self.spike_excitatory
            self.time += self.dt
            print(f'time: {self.time:.1f} ms', end='\r')

        return spike_record_input, spike_record_excitatory

    def label_neurons(self, train_data, num_classes=10, STP_on=True):
        self.classes = num_classes
        num_neurons = self.layers['excitatory'].num_neuron
        response = torch.zeros((num_neurons, num_classes))
        label_count = torch.zeros((num_classes,))

        for img, label in train_data:
            _, spike_record = self.run(input_data=img, simtime=350.0, train=False, STP_on=STP_on)
            spike_count = spike_record.sum(dim=1)
            response[:, label] += spike_count
            label_count[label] += 1

            self.run(train=False, simtime=150.0, STP_on=STP_on)

        for c in range(num_classes):
            if label_count[c] > 0:
                response[:, c] /= label_count[c]

        self.preferred_label = torch.argmax(response, dim=1)

        return self.preferred_label
    
    def get_label(self, input_data):
        _, spike_record = self.run(input_data=input_data, simtime=350.0, train=False)
        spike_count = spike_record.sum(dim=1)
        if self.preferred_label is None:
            raise ValueError("Neurons have not been labeled yet. Please run label_neurons() first.")
        if spike_count.sum() == 0:
            return None
        votes = torch.zeros((self.classes,), device=self.device)
        for label in range(self.classes):
            votes[label] = spike_count[self.preferred_label == label].sum()

        return torch.argmax(votes).item()


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

    print("训练网络中（含STDP学习）...")
    for img, label in train_data:
        model.run(input_data=img, simtime=350.0, train=True)
        model.run(train=False, simtime=150.0)

    print("训练完成！")

    print("根据神经元响应打标签...")
    preferred_label = model.label_neurons(train_data, num_classes=10)
    print("标注完成！")

    print("神经元的首选标签（前30个神经元）：")
    print(preferred_label[:30])
    label_count = pd.Series(preferred_label.numpy()).value_counts().sort_index()
    print("\n每类神经元数量分布：")
    print(label_count)

def get_mnist_data(batch_size=1, n_per_class=10, train=True):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
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

    train_loader = get_mnist_data(batch_size=1, n_per_class=3, train=True)

    print("训练网络中（含STDP学习）...")
    for (img, label) in list(train_loader)[:20]:
        img, label = img.squeeze(0), label.item()
        model.run(input_data=img, simtime=350.0, train=True, STP_on=False)
        model.run(train=False, simtime=150.0)

    print("训练完成！")

    print("根据神经元响应打标签...")
    mnist_samples = [(img.squeeze(0), label.item()) for img, label in list(train_loader)[:20]]
    preferred_label = model.label_neurons(mnist_samples, num_classes=10, STP_on=False)
    print("标注完成！")

    print("神经元的首选标签（前30个神经元）：")
    print(preferred_label[:30])

    label_count = pd.Series(preferred_label.cpu().numpy()).value_counts().sort_index()
    print("\n每类神经元数量分布：")
    for i, count in label_count.items():
        print(f"{i}: {count}")

        print("\n测试阶段（预测准确率）:")
    correct, total = 0, 0
    for (img, label) in list(train_loader)[20:]:
        img = img.squeeze(0)
        pred = model.get_label(img)
        if pred is not None and pred == label.item():
            correct += 1
        total += 1

    print(f"预测准确率: {correct}/{total} = {100 * correct / total:.2f}%")


if __name__ == "__main__":
    mnist_test()
