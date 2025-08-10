import torch
import numpy as np
import matplotlib.pyplot as plt
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
from legendre import *
import os

nu=1e-2
nu_f = '1e-2'
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

        self.X_mean = torch.from_numpy(X.mean(0, keepdims=True)).float()
        self.X_std = torch.from_numpy(X.std(0, keepdims=True)).float()
        self.X_mean = self.X_mean.to(device)
        self.X_std = self.X_std.to(device)

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



def process_nu_and_plot(nu_mapping, base_path, save_path, device='cpu'):
    """
    Process multiple `nu` values and generate plots for each.

    Parameters:
        nu_mapping (dict): A dictionary mapping `nu` (float) to its corresponding formatted string `nu_f`.
        base_path (str): Path to the folder containing models and data.
        save_path (str): Path to save the plots.
        device (str): Device to run the models on ('cpu' or 'cuda').
    """
    for nu, nu_f in nu_mapping.items():
        # Load models
        model1_path = f"{base_path}/mlp_{nu}.pth"
        model2_path = f"{base_path}/legend_{nu}.pth"
        data_filename = f"{data_path}/cylinder_{nu_f}mu.npy"
        print(model1_path)
        print(model2_path)

        model1 = torch.load(model1_path, map_location=torch.device(device),weights_only=False)
        model2 = torch.load(model2_path, map_location=torch.device(device),weights_only=False)
        model1.eval()
        model2.eval()

        # Load data
        Data = np.load(data_filename)
        x, y = Data[:, 0], Data[:, 1]
        u_true, v_true = Data[:, 2], Data[:, 3]
        U_true = np.sqrt(u_true**2 + v_true**2)

        # Prepare model input
        data = np.column_stack((x, y))
        model_input = torch.tensor(data[:, :2]).float().to(device)
        # nu_tensor = nu * torch.ones((model_input.shape[0], 1), device=device)
        # nu_tensor = torch.log(nu_tensor)
        # model_input = torch.cat([model_input, nu_tensor], dim=1).float()

        # Model predictions
        model_output1 = model1(model_input)
        model_output2 = model2(model_input)

        u_pred1, v_pred1 = model_output1[:, 0].cpu().detach().numpy(), model_output1[:, 1].cpu().detach().numpy()
        u_pred2, v_pred2 = model_output2[:, 0].cpu().detach().numpy(), model_output2[:, 1].cpu().detach().numpy()

        U_pred1 = np.sqrt(u_pred1**2 + v_pred1**2)
        U_pred2 = np.sqrt(u_pred2**2 + v_pred2**2)

        # Compute absolute errors
        absolute_error1 = np.abs(U_true - U_pred1)
        absolute_error2 = np.abs(U_true - U_pred2)
        # 计算相对L2误差
        l2_error1 = np.linalg.norm(U_true - U_pred1) / np.linalg.norm(U_true)
        l2_error2 = np.linalg.norm(U_true - U_pred2) / np.linalg.norm(U_true)

        # Visualize and save
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        error_min = min(absolute_error1.min(), absolute_error2.min())
        error_max = max(absolute_error1.max(), absolute_error2.max())

        # Real value plot
        axs[0].scatter(x, y, c=U_true, cmap='jet', s=5)
        axs[0].set_title('True U')
        axs[0].set_xlim(x_min, x_max)
        axs[0].set_ylim(y_min, y_max)
        axs[0].set_aspect('equal')

        # Model 1 prediction
        axs[1].scatter(x, y, c=U_pred1, cmap='jet', s=5)
        axs[1].set_title('LegendKINN Prediction')
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(y_min, y_max)
        axs[1].set_aspect('equal')

        # Model 1 error
        scatter3 = axs[2].scatter(x, y, c=absolute_error1, cmap='jet', s=5, vmin=error_min, vmax=error_max)
        axs[2].set_title('MLP Error')
        axs[2].set_xlim(x_min, x_max)
        axs[2].set_ylim(y_min, y_max)
        axs[2].set_aspect('equal')
        fig.colorbar(scatter3, ax=axs[2], shrink=0.4, pad=0.001)

        # Model 2 prediction
        axs[3].scatter(x, y, c=U_pred2, cmap='jet', s=5)
        axs[3].set_title('LegendKINN Prediction')
        axs[3].set_xlim(x_min, x_max)
        axs[3].set_ylim(y_min, y_max)
        axs[3].set_aspect('equal')

        # Model 2 error
        scatter5 = axs[4].scatter(x, y, c=absolute_error2, cmap='jet', s=5, vmin=error_min, vmax=error_max)
        axs[4].set_title('LegendKINN Error')
        axs[4].set_xlim(x_min, x_max)
        axs[4].set_ylim(y_min, y_max)
        axs[4].set_aspect('equal')
        fig.colorbar(scatter5, ax=axs[4], shrink=0.4, pad=0.001)

        # Save plot
        plt.savefig(f"{save_path}/_{nu_f}_mlp.vs.legend.png", dpi=600)
        plt.clf()

        # Calculate and print average errors
        avg_error1 = np.mean(absolute_error1)
        avg_error2 = np.mean(absolute_error2)


        print(f"nu = {nu_f}: MLP Average Absolute Error = {avg_error1:.6f}")
        print(f"nu = {nu_f}: LegendKINN Average Absolute Error = {avg_error2:.6f}")

        print(f"nu = {nu_f}: MLP Relative L2 Error = {l2_error1:.6f}")
        print(f"nu = {nu_f}: LegendKINN Relative L2 Error = {l2_error2:.6f}")



# Example usage
nu_mapping = {
    1.25e-2: '1.25e-2',
    1.25e-3: '1.25e-3',
    1e-2: '1e-2',
    2.5e-3: '2.5e-3',
    2e-2: '2e-2',
    2e-3: '2e-3',
    5e-3: '5e-3'

}
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, "checkpoint")
data_path = os.path.join(script_dir, "data")
save_path = os.path.join(script_dir, "vis_result")

process_nu_and_plot(nu_mapping, base_path, save_path, device='cpu')
