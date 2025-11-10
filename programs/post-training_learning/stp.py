import torch
import pandas as pd

class lif:
    def __init__(self, num_neuron=400, E_exc=0.0, E_inh=-100.0, E_rest=-65.0, V_thr=52.0, tau=100.0, tau_ref=2.0, tau_e=2.0, tau_i=1.0, dt=0.1):
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
        self.V = torch.full((num_neuron,), E_rest)
        self.last_spike_time = torch.full((num_neuron,), -float('inf'))
        self.g_e = torch.full((num_neuron,), 0.0)
        self.g_i = torch.full((num_neuron,), 0.0)

    def reset(self):
        self.V = torch.full((self.num_neuron,), self.E_rest)
        self.last_spike_time = torch.full((self.num_neuron,), -float('inf'))

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
    def __init__(self, omega_d=2.0, omega_f=3.33, U_0=0.6, w_ei=10.5, w_ie=17.0, lr_pre=1e-4, lr_post=1e-2, tau_post1=1.0, tau_post2=2.0, tau_pre=1.5, k=0.6, input_size=28*28, dt=0.1):
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

        input_layer = lif(num_neuron=input_size, dt=dt)
        excitatory_layer = lif(num_neuron=400, dt=dt)
        inhibitory_layer = lif(num_neuron=400, E_exc=0.0, E_inh=-85.0, E_rest=-60.0, V_thr=-40.0, tau=10.0, tau_ref=5.0, tau_e=2.0, tau_i=1.0, dt=dt)
        self.layers = {
            'input': input_layer,
            'excitatory': excitatory_layer,
            'inhibitory': inhibitory_layer
        }

        self.a_pre = torch.zeros_like(self.layers['input'].V)
        self.a_post1 = torch.zeros_like(self.layers['excitatory'].V)
        self.a_post2 = torch.zeros_like(self.layers['excitatory'].V)
        self.time = 0.0
        self.w = torch.randn((input_size, 400)) * 0.1
        self.u_input = torch.full((input_size,), 0.0)
        self.x_input = torch.full((input_size,), 1.0)
        self.u_excitatory = torch.full((400,), 0.0)
        self.x_excitatory = torch.full((400,), 1.0)
        self.u_inhibitory = torch.full((400,), 0.0)
        self.x_inhibitory = torch.full((400,), 1.0)
        self.spike_input = torch.zeros((input_size,), dtype=torch.bool)
        self.spike_excitatory = torch.zeros((400,), dtype=torch.bool)
        self.spike_inhibitory = torch.zeros((400,), dtype=torch.bool)

    def reset(self):
        for layer in self.layers.values():
            layer.reset()
        self.time = 0.0
        self.u_input = torch.zeros_like(self.layers['input'].V)
        self.x_input = torch.ones_like(self.layers['input'].V)
        self.u_excitatory = torch.zeros_like(self.layers['excitatory'].V)
        self.x_excitatory = torch.ones_like(self.layers['excitatory'].V)
        self.u_inhibitory = torch.zeros_like(self.layers['inhibitory'].V)
        self.x_inhibitory = torch.ones_like(self.layers['inhibitory'].V)
        self.spike_input = torch.zeros_like(self.layers['input'].V, dtype=torch.bool)
        self.spike_excitatory = torch.zeros_like(self.layers['excitatory'].V, dtype=torch.bool)
        self.spike_inhibitory = torch.zeros_like(self.layers['inhibitory'].V, dtype=torch.bool)

    def STDP(self):
        self.a_pre -= self.a_pre * self.dt / self.tau_pre
        self.a_post1 -= self.a_post1 * self.dt / self.tau_post1
        self.a_post2 -= self.a_post2 * self.dt / self.tau_post2

        if self.spike_input.any():
            self.a_pre[self.spike_input] = 1.0
            dw = self.lr_pre * self.a_post1.unsqueeze(0)
            self.w[self.spike_input, :] += dw

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
    
    def run(self, input_data=None, simtime=1000.0, train=True):
        num_steps = int(simtime / self.dt)
        spike_record_excitatory = torch.zeros((self.layers['excitatory'].num_neuron, num_steps), dtype=torch.bool)
        t = 0.0
        for step in range(num_steps):
            if input_data is None:
                input_data = torch.zeros((self.layers['input'].num_neuron,), dtype=torch.bool)

            self.spike_input = self.layers['input'].forward(w_e=input_data, 
                                                            w_i=torch.zeros((self.layers['input'].num_neuron,), dtype=torch.float32),
                                                            t=t)
            r_input = self.STP()
            w_effective = self.k * self.w * r_input.unsqueeze(1)
            self.spike_excitatory = self.layers['excitatory'].forward((self.w + w_effective).sum(dim=0), 
                                                                      torch.zeros((400,), dtype=torch.float32), 
                                                                      t)
            self.spike_inhibitory = self.layers['inhibitory'].forward(self.w_ei * self.spike_excitatory.float(), 
                                                                      torch.zeros((400,), dtype=torch.float32), 
                                                                      t)
            self.layers['excitatory'].update_g(w_e=torch.zeros((400,), dtype=torch.float32), 
                                               w_i=self.w_ie * (torch.full_like(self.spike_inhibitory.float(), self.spike_inhibitory.sum()) - self.spike_inhibitory.float()), 
                                               is_feedback=True)
            if train:
                self.STDP()
            spike_record_excitatory[:, step] = self.spike_excitatory
            t += self.dt

        return spike_record_excitatory

    def label_neurons(self, train_data, num_classes=10):
        num_neurons = self.layers['excitatory'].num_neuron
        response = torch.zeros((num_neurons, num_classes))

        for img, label in train_data:
            spike_record = self.run(input_data=img, simtime=500.0, train=False)
            spike_count = spike_record.sum(dim=1)
            response[:, label] += spike_count

            self.run(train=False, simtime=100.0)

        preferred_label = torch.argmax(response, dim=1)

        return preferred_label
