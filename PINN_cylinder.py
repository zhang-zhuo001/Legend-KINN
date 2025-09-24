import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.nn.utils.weight_norm as weight_norm
import time
import pickle
import copy

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph=True)[0]
    return G

class Net(torch.nn.Module):
    def __init__(self, layer_dim, X, device):
        super().__init__()
        self.num_layers = len(layer_dim)
        temp = []
        for l in range(1, self.num_layers):
            temp.append(weight_norm(torch.nn.Linear(layer_dim[l - 1], layer_dim[l]), dim=0))
            torch.nn.init.xavier_normal_(temp[l - 1].weight)
        self.layers = torch.nn.ModuleList(temp)
        self.layers.append(torch.nn.BatchNorm1d(layer_dim[l]))

    def forward(self, x):
        for i in range(0, self.num_layers - 1):

            x = self.layers[i](x)
            if i < self.num_layers - 2:
                x = torch.tanh(x)
        return x

class TSONN():

    def __init__(self, layers, device):
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        self.layers = layers
        self.device = device
        points = np.load(data_dir)
        INNER = points['INNER']
        INLET = points['INLET']
        OUTLET = points['OUTLET']
        UP = points['UP']
        DOWN = points['DOWN']
        CIR = points['CIR']
        U0 = points['U0']
        WALL = np.concatenate((UP, DOWN, CIR), 0)
        cy_d=1
        x_inner = torch.tensor(INNER[:, 0:1]/cy_d)
        y_inner = torch.tensor(INNER[:, 1:2]/cy_d)
        # self.INNER_dataset = Data.TensorDataset(x_inner, y_inner)
        self.x = torch.tensor(INNER[:, 0:1]/cy_d, requires_grad=True).to(device)
        self.y = torch.tensor(INNER[:, 1:2]/cy_d, requires_grad=True).to(device)
        self.X = torch.cat([self.x, self.y], dim=1)
        # print(self.X.shape)
        self.x_INLET = torch.tensor(INLET[:, 0:1]/cy_d, requires_grad=True).to(device)
        self.y_INLET = torch.tensor(INLET[:, 1:2]/cy_d, requires_grad=True).to(device)
        self.X_INLET = torch.cat([self.x_INLET, self.y_INLET], dim=1).float()
        # print(self.X_INLET.shape)
        self.u_INLET = torch.tensor(U0[:, 0:1]/cy_d).to(device)
        self.v_INLET = torch.tensor(U0[:, 1:2]/cy_d).to(device)

        self.x_OUTLET = torch.tensor(OUTLET[:, 0:1]/cy_d, requires_grad=True).to(device)
        self.y_OUTLET = torch.tensor(OUTLET[:, 1:2]/cy_d, requires_grad=True).to(device)
        self.X_OUTLET = torch.cat([self.x_OUTLET, self.y_OUTLET], dim=1).float()

        self.x_WALL = torch.tensor(WALL[:, 0:1]/cy_d, requires_grad=True).to(device)
        self.y_WALL = torch.tensor(WALL[:, 1:2]/cy_d, requires_grad=True).to(device)
        self.X_WALL = torch.cat([self.x_WALL, self.y_WALL], dim=1).float()

        self.Nx = self.Ny = 501

        self.min_loss = 1
        self.log = {'losses': [], 'losses_b': [], 'losses_i': [], 'losses_f': [], 'losses_s': [], 'time': []}

        self.model = Net(self.layers, self.X.cpu().detach().numpy(),self.device).to(self.device)

    def Mseb(self):

        Y_INLET_pred = self.model(self.X_INLET)
        Y_OUTLET_pred = self.model(self.X_OUTLET)
        Y_WALL_pred = self.model(self.X_WALL)
        u_INLET_pred = Y_INLET_pred[:, 0:1]
        v_INLET_pred = Y_INLET_pred[:, 1:2]
        p_OUTLET_pred = Y_OUTLET_pred[:, 2:3]
        u_WALL_pred = Y_WALL_pred[:, 0:1]
        v_WALL_pred = Y_WALL_pred[:, 1:2]



        loss_WALL = self.loss_fn(u_WALL_pred.float(), torch.zeros_like(u_WALL_pred).float()) + \
                    self.loss_fn(v_WALL_pred.float(), torch.zeros_like(v_WALL_pred).float())

        loss_INLET = self.loss_fn(u_INLET_pred.float(), self.u_INLET.float()) + \
                     self.loss_fn(v_INLET_pred.float(), self.v_INLET.float())

        loss_OUTLET = self.loss_fn(p_OUTLET_pred.float(), torch.zeros_like(p_OUTLET_pred).float())

        loss = loss_WALL+ loss_INLET + loss_OUTLET
        return loss



    def TimeStepping(self):
        X = self.X
        X = X.float()

        pred = self.model(X)
        u = pred[:, 0:1];
        v = pred[:, 1:2];
        p = pred[:, 2:3];

        self.U0 = torch.cat([u, v, p]).detach()

    def Msef(self):
        X = self.X
        X = X.float()
        pred = self.model(X)
        # print(pred.dtype)
        u = pred[:, 0:1];
        # print(u.dtype)
        v = pred[:, 1:2];
        # print(v.dtype)
        p = pred[:, 2:3];
        # print(p.dtype)

        u_xy = fwd_gradients(u, X)
        v_xy = fwd_gradients(v, X)
        p_xy = fwd_gradients(p, X)
        u_x = u_xy[:, 0:1];
        u_y = u_xy[:, 1:2]
        v_x = v_xy[:, 0:1];
        v_y = v_xy[:, 1:2]
        p_x = p_xy[:, 0:1];
        p_y = p_xy[:, 1:2]

        u_xx = fwd_gradients(u_x, X)[:, 0:1]
        u_yy = fwd_gradients(u_y, X)[:, 1:2]
        v_xx = fwd_gradients(v_x, X)[:, 0:1]
        v_yy = fwd_gradients(v_y, X)[:, 1:2]

        res_rho = u_x + v_y
        # res_u = u * u_x + v * u_y + p_x - 1 / self.Re * (u_xx + u_yy)
        # res_v = u * v_x + v * v_y + p_y - 1 / self.Re * (v_xx + v_yy)
        res_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        res_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        msef = (res_u ** 2 + res_v ** 2 + res_rho ** 2).mean()

        U1 = torch.cat([u, v, p])
        R1 = torch.cat([res_u, res_v, res_rho])
        msef = (R1**2).mean()

        dtau = 0.5
        msef = 1 / dtau ** 2 * ((U1 - self.U0 + dtau * R1) ** 2).mean()
        return msef

    def Mses(self):
        X_ref = self.X_ref
        X_ref = X_ref.float()
        pred = self.model(X_ref)
        u = pred[:, 0:1]
        v = pred[:, 1:2]
        V = torch.sqrt(u ** 2 + v ** 2).detach()
        mses = torch.norm(V - self.V_ref, p=2) / torch.norm(self.V_ref, p=2)
        return mses


    def Loss(self):
        msef = self.Msef()
        mseb = self.Mseb()

        mseb = mseb
        msef = msef

        loss = mseb + msef
        # print(f"Loss dtype: {loss.dtype}")
        return loss, mseb, msef

    def train(self, adam_epochs):
        print(f"Loss function initialized: {self.loss_fn}")

        if len(self.log['time']) == 0:
            t1 = time.time()
        else:
            t1 = time.time() - self.log['time'][-1]

        learning_rates = [1e-3, 1e-4]
        lr_epochs = adam_epochs // len(learning_rates)

        # Open log file and write the header
        with open(f'result/{nu_f}_mlp.csv', 'a') as log_file:
            log_file.write('Epoch,Loss,Loss_f,Loss_s,time\n')

            # Adam pre-training with staged learning rates
            for stage, lr in enumerate(learning_rates):
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                start_epoch = stage * lr_epochs
                end_epoch = start_epoch + lr_epochs

                for i in range(start_epoch, min(end_epoch, adam_epochs)):
                    self.TimeStepping()
                    self.optimizer.zero_grad()
                    self.loss, self.loss_b, self.loss_f = self.Loss()
                    self.loss.backward()
                    self.optimizer.step()

                    # Calculate loss_s and log it
                    self.loss_s = self.Mses()
                    t2 = time.time()

                    # Log losses and time
                    self.log['losses'].append(self.loss.item())
                    self.log['losses_f'].append(self.loss_f.item())
                    self.log['losses_b'].append(self.loss_b.item())
                    self.log['losses_s'].append(self.loss_s.item())
                    self.log['time'].append(t2 - t1)
                    log_file.write(f'{i},{self.loss.item()},{self.loss_f.item()},{self.loss_s.item()},{t2 - t1}\n')

                    # Print loss every 100 iterations
                    if i % 10 == 0:
                        print(
                            f'Adam Pretrain {i}/{adam_epochs} - LR: {lr} - Loss: {self.loss.item()} error={self.loss_s}')

                    if i % 1000 == 0:
                        torch.save(self.model, f'checkpoint/mlp_{nu_f}.pth')

if __name__ == '__main__':
    t1 = time.time()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    adam_epochs=100000
    l_epochs=100
    nu_values = {
        1.25e-2: '1.25e-2',
        1.25e-3: '1.25e-3',
        1e-2: '1e-2',
        2.5e-3: '2.5e-3',
        2e-2: '2e-2',
        2e-3: '2e-3',
        5e-3: '5e-3'

    }

    for nu_f, nu_str in nu_values.items():
        print(nu_str)
        print(nu_f)
        nu = torch.tensor(nu_f).to(device)

        data_dir = f'cylinder_points.npz'
        data_filename = f'cylinder_{nu_str}mu.npy'

        Data = np.load(data_filename)

        layers = [2, 20, 20, 20, 3]
        nn = TSONN(layers, device)

        x = Data[:, 0]
        y = Data[:, 1]
        data = np.column_stack((x, y))

        nn.X_ref = torch.tensor(data[:, :2]).to(device)
        nn.V_ref = torch.tensor(Data[:, 4:5]).to(device)

        print(f'Training with nu = {nu_str}')
        nn.train(adam_epochs)

    t2 = time.time()
    print('Total time: ', t2 - t1)
