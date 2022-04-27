import numpy as np
import equation_discovery as ed
import van_der_pol as vdp
import lorenz
import torch.nn.utils.prune
import torch
from torch.autograd import Variable
import h5py
import pickle as pk
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.utils.extmath import cartesian
from torch.utils.data import Dataset, DataLoader, IterableDataset
from skimage.measure import block_reduce
import scipy.io
from scipy.ndimage import convolve
from skimage.filters import gaussian
from itertools import product
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

from nngeometry.generator import Jacobian
from nngeometry.object import FMatDense, PMatDense
from nngeometry.metrics import FIM
import pickle as pk
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset

def seed_all(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)

def main_burger():
    f = scipy.io.loadmat('burgers.mat')
    u, x, t = np.real(f['usol']), f['x'], f['t']
    x = np.reshape(x, (x.size, 1))
    return u,x[:, 0],t[:,0]

class DataVDP(Dataset):
    def __init__(self, synth_args = None, subsample_zero = 0.05, subsample_all = 1, smoothing = 0.0, already_generated = None, subsample_base = 6):
        self.synth_args = synth_args
        self.subsample_zero = subsample_zero
        self.subsample_all = subsample_all
        if already_generated is None:
          self.generate_data()
        else:
          self.u = already_generated['u']
          self.x = already_generated['x']
          self.y = already_generated['y']
          self.t = already_generated['t']
        self.correct_coords()
        self.xyt = cartesian((self.x, self.y, self.t))
        self.u_base = np.zeros_like(self.u)
        for i in range(len(self.t)):
            self.u_base[:,:,i] = gaussian(self.u[:,:,i], sigma = subsample_base)
        self.u_base_flat = self.u_base.flatten()
        self.get_subsample()
        if smoothing > 0:
          for i in range(len(self.t)):
            self.u[:,:,i] = gaussian(self.u[:,:,i], sigma = smoothing)
        self.u_flat = self.u.flatten()

    def correct_coords(self):
        x_range, y_range = self.synth_args['x_range'], self.synth_args['y_range']
        num_x, num_y = self.synth_args['num_x'], self.synth_args['num_y']
        x_off = (x_range[1] - x_range[0])/(2*num_x)
        y_off = (y_range[1] - y_range[0])/(2*num_y)
        self.x = np.linspace(x_range[0]+x_off, x_range[1]-x_off, num_x)
        self.y = np.linspace(y_range[0]+y_off, y_range[1]-y_off, num_y)
    def generate_data(self):
        x_range, y_range = self.synth_args['x_range'], self.synth_args['y_range']
        num_x, num_y = self.synth_args['num_x'], self.synth_args['num_y']
        t_max, dt = self.synth_args['t_max'], self.synth_args['dt']
        mu, n, r = self.synth_args['mu'], self.synth_args['n'], self.synth_args['r']
        bandwidth = self.synth_args['bandwidth']
        ic_mode  =self.synth_args['ic_mode']
        self.u = vdp.van_der_pol(x_range, y_range, num_x, num_y, t_max, dt, mu, n, r, bandwidth=bandwidth, ic_mode = ic_mode)
        self.u = np.transpose(self.u, (1,2,0)) #shape (x,y,t)
        self.x = np.linspace(x_range[0], x_range[1], num_x)
        self.y = np.linspace(y_range[0], y_range[1], num_y)
        self.t = np.linspace(0, t_max, int(t_max/dt) + 1)
    def get_subsample(self):
        zero_idx = np.where(self.u_base_flat == 0)[0]
        nonz_idx = np.where(self.u_base_flat != 0)[0]
        subsamples = np.random.choice(zero_idx, size = int(self.subsample_zero * len(zero_idx)), replace=False)
        self.samples = np.concatenate([subsamples, nonz_idx])
        self.samples = np.random.choice(self.samples, size = int(len(self.samples) * self.subsample_all), replace = False)
    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        return self.xyt[real_idx], self.u_flat[real_idx]

    def __len__(self):
        return self.samples.size

class DataLorenz(Dataset):
    def __init__(self, synth_args = None, subsample_zero = 0.05, subsample_all = 1, smoothing = 0.0, already_generated = None, subsample_base = 6):
        self.synth_args = synth_args
        self.subsample_zero = subsample_zero
        self.subsample_all = subsample_all
        if already_generated is None:
          self.generate_data()
        else:
          self.u = already_generated['u']
          self.x = already_generated['x']
          self.y = already_generated['y']
          self.z = already_generated['z']
          self.t = already_generated['t']
        self.u_base = np.zeros_like(self.u)
        for i in range(len(self.t)):
            self.u_base[:,:,:, i] = gaussian(self.u[:, :, :, i], sigma = subsample_base)
        self.u_base = self.u_base.flatten()
        self.get_subsample()
        del self.u_base
        if smoothing > 0:
          for i in range(len(self.t)):
            self.u[:,:,:, i] = gaussian(self.u[:, :, :, i], sigma = smoothing)
        self.u_flat = self.u.flatten()
        self.correct_coords()

    def correct_coords(self):
        x_range, y_range, z_range = self.synth_args['x_range'], self.synth_args['y_range'], self.synth_args['z_range']
        num_x, num_y, num_z = self.synth_args['num_x'], self.synth_args['num_y'], self.synth_args['num_z']
        x_off = (x_range[1] - x_range[0]) / (2 * num_x)
        y_off = (y_range[1] - y_range[0]) / (2 * num_y)
        z_off = (z_range[1] - z_range[0]) / (2 * num_z)
        self.x = np.linspace(x_range[0] + x_off, x_range[1] - x_off, num_x)
        self.y = np.linspace(y_range[0] + y_off, y_range[1] - y_off, num_y)
        self.z = np.linspace(z_range[0] + z_off, z_range[1] - z_off, num_z)

    def generate_data(self):
        x_range, y_range, z_range = self.synth_args['x_range'], self.synth_args['y_range'], self.synth_args['z_range']
        num_x, num_y,num_z = self.synth_args['num_x'], self.synth_args['num_y'],self.synth_args['num_z']
        t_max, dt = self.synth_args['t_max'], self.synth_args['dt']
        n = self.synth_args['n']
        spherical_r = self.synth_args['spherical_r']
        self.u = lorenz.lorenz(x_range, y_range,z_range, num_x, num_y,num_z, t_max, dt, n, spherical_r)
        self.u = np.transpose(self.u, (1,2,3,0)) #shape (x,y,z,t)
        self.x = np.linspace(x_range[0], x_range[1], num_x)
        self.y = np.linspace(y_range[0], y_range[1], num_y)
        self.z = np.linspace(z_range[0], z_range[1], num_z)
        self.t = np.linspace(0, t_max, int(t_max/dt) + 1)
    def get_subsample(self):
        zero_idx = np.where(self.u_base == 0)[0]
        nonz_idx = np.where(self.u_base != 0)[0]
        subsamples = np.random.choice(zero_idx, size = int(self.subsample_zero * len(zero_idx)), replace=False)
        self.samples = np.concatenate([subsamples, nonz_idx])
        self.samples = np.random.choice(self.samples, size = int(len(self.samples) * self.subsample_all), replace = False)
    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        x_idx = real_idx // (len(self.y) * len(self.z) * len(self.t))
        y_idx = (real_idx // (len(self.z) * len(self.t))) % len(self.y)
        z_idx = (real_idx // len(self.t)) % len(self.z)
        t_idx = real_idx % len(self.t)
        return np.array([self.x[x_idx],self.y[y_idx],self.z[z_idx],self.t[t_idx]]), self.u_flat[real_idx]

    def __len__(self):
        return self.samples.size

class DataVDPST(Dataset):
    def __init__(self, synth_args = None, smoothing = 0.0, already_generated = None, st_subsample= 2e-3, st_sub_sig = 10):
        self.synth_args = synth_args
        self.st_subsample = st_subsample
        self.st_sub_sig = st_sub_sig
        if already_generated is None:
          self.generate_data()
        else:
          self.u = already_generated['u']
          self.x = already_generated['x']
          self.y = already_generated['y']
          self.t = already_generated['t']
        if smoothing > 0:
          for i in range(len(self.t)):
            self.u[:,:,i] = gaussian(self.u[:,:,i], sigma = smoothing)
        self.correct_coords()
        self.get_subsample()

    def correct_coords(self):
        x_range, y_range = self.synth_args['x_range'], self.synth_args['y_range']
        num_x, num_y = self.synth_args['num_x'], self.synth_args['num_y']
        x_off = (x_range[1] - x_range[0])/(2*num_x)
        y_off = (y_range[1] - y_range[0])/(2*num_y)
        self.x = np.linspace(x_range[0]+x_off, x_range[1]-x_off, num_x)
        self.y = np.linspace(y_range[0]+y_off, y_range[1]-y_off, num_y)

    def generate_data(self):
        x_range, y_range = self.synth_args['x_range'], self.synth_args['y_range']
        num_x, num_y = self.synth_args['num_x'], self.synth_args['num_y']
        t_max, dt = self.synth_args['t_max'], self.synth_args['dt']
        mu, n, r = self.synth_args['mu'], self.synth_args['n'], self.synth_args['r']
        bandwidth = self.synth_args['bandwidth']
        ic_mode  =self.synth_args['ic_mode']
        self.u = vdp.van_der_pol(x_range, y_range, num_x, num_y, t_max, dt, mu, n, r, bandwidth=bandwidth, ic_mode = ic_mode)
        self.u = np.transpose(self.u, (1,2,0)) #shape (x,y,t)
        self.x = np.linspace(x_range[0], x_range[1], num_x)
        self.y = np.linspace(y_range[0], y_range[1], num_y)
        self.t = np.linspace(0, t_max, int(t_max/dt) + 1)

    def get_subsample(self):
        nx, ny = len(self.x), len(self.y)
        p = np.zeros((nx, ny))
        p[[nx // 2 - 1, nx // 2], [ny // 2 - 1, ny // 2]] = 1 / 2.0
        p = gaussian(p, sigma=self.st_sub_sig)
        idxes = cartesian((np.arange(nx), np.arange(ny)))
        sub_xy = np.sort(
            np.random.choice(len(idxes), size=int(self.st_subsample * idxes.size), p=p.flatten(), replace=False))
        xy = cartesian([self.x, self.y])[sub_xy]
        xy = np.repeat(xy, len(self.t), axis=0)
        t_tile = np.tile(self.t, len(sub_xy))
        t_tile = np.reshape(t_tile, (-1, 1))
        self.xyt = np.hstack([xy, t_tile])
        self.u = np.reshape(self.u, (-1, len(self.t)))
        self.u = self.u[sub_xy, :]
        self.u = self.u.flatten()

    def __getitem__(self, idx):
        return self.xyt[idx], self.u[idx]

    def __len__(self):
        return self.u.size


class DataLorenzST(Dataset):
    def __init__(self, synth_args = None, smoothing = 0.0, already_generated = None, st_subsample= 2e-3, st_sub_sig = 10):
        self.synth_args = synth_args
        self.st_subsample = st_subsample
        self.st_sub_sig = st_sub_sig
        if already_generated is None:
          self.generate_data()
        else:
          self.u = already_generated['u']
          self.x = already_generated['x']
          self.y = already_generated['y']
          self.z = already_generated['z']
          self.t = already_generated['t']
        if smoothing > 0:
          for i in range(len(self.t)):
            self.u[:,:,:, i] = gaussian(self.u[:, :, :, i], sigma = smoothing)
        self.correct_coords()
        self.get_subsample()

    def correct_coords(self):
        x_range, y_range, z_range = self.synth_args['x_range'], self.synth_args['y_range'], self.synth_args['z_range']
        num_x, num_y, num_z = self.synth_args['num_x'], self.synth_args['num_y'], self.synth_args['num_z']
        x_off = (x_range[1] - x_range[0]) / (2 * num_x)
        y_off = (y_range[1] - y_range[0]) / (2 * num_y)
        z_off = (z_range[1] - z_range[0]) / (2 * num_z)
        self.x = np.linspace(x_range[0] + x_off, x_range[1] - x_off, num_x)
        self.y = np.linspace(y_range[0] + y_off, y_range[1] - y_off, num_y)
        self.z = np.linspace(z_range[0] + z_off, z_range[1] - z_off, num_z)

    def generate_data(self):
        x_range, y_range, z_range = self.synth_args['x_range'], self.synth_args['y_range'], self.synth_args['z_range']
        num_x, num_y,num_z = self.synth_args['num_x'], self.synth_args['num_y'],self.synth_args['num_z']
        t_max, dt = self.synth_args['t_max'], self.synth_args['dt']
        n = self.synth_args['n']
        spherical_r = self.synth_args['spherical_r']
        self.u = lorenz.lorenz(x_range, y_range,z_range, num_x, num_y,num_z, t_max, dt, n, spherical_r)
        self.u = np.transpose(self.u, (1,2,3,0)) #shape (x,y,z,t)
        self.x = np.linspace(x_range[0], x_range[1], num_x)
        self.y = np.linspace(y_range[0], y_range[1], num_y)
        self.z = np.linspace(z_range[0], z_range[1], num_z)
        self.t = np.linspace(0, t_max, int(t_max/dt) + 1)
    def get_subsample(self):
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        p = np.zeros((nx, ny, nz))
        p[[nx//2 - 1, nx //2], [ny //2 - 1, ny//2], [nz//2 - 1, nz //2 ]] = 1 / 2.0
        p = gaussian(p, sigma=self.st_sub_sig)
        idxes = cartesian((np.arange(nx), np.arange(ny), np.arange(nz)))
        sub_xyz = np.sort(np.random.choice(len(idxes), size=int(self.st_subsample * idxes.size), p=p.flatten(), replace=False))
        xyz = cartesian([self.x, self.y, self.z])[sub_xyz]
        xyz = np.repeat(xyz, len(self.t), axis = 0)
        t_tile = np.tile(self.t, len(sub_xyz))
        t_tile = np.reshape(t_tile, (-1, 1))
        self.xyzt = np.hstack([xyz, t_tile])
        self.u = np.reshape(self.u, (-1, len(self.t)))
        self.u = self.u[sub_xyz, :]
        self.u = self.u.flatten()
    def __getitem__(self, idx):
        return self.xyzt[idx], self.u[idx]
    def __len__(self):
        return self.xyzt.size

class NetDiff3DAdvection(pl.LightningModule):
    def __init__(self, actv, num_inputs, hidden_units, num_outputs, lr=1e-3, lam=1e-4, alpha=1, device='cpu',
                 schedule_reg_epoch=0):
        super(NetDiff3DAdvection, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.schedule_reg_epoch = schedule_reg_epoch
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.lr = lr
        self.lam = lam
        self.device_name = device

        # Assign activation function (exec allows us to assign function from string)
        exec('self.actv = %s' % actv)

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Loop over layers and create each one
        for i in range(len(hidden_units)):
            next_num_inputs = hidden_units[i]
            self.layers += [nn.Linear(num_inputs, next_num_inputs)]
            num_inputs = next_num_inputs

        # Create final layer
        self.layers += [nn.Linear(num_inputs, num_outputs)]

        poly_desc = ['1', 'x', 'x^2', 'x^3', 'y', 'y^2', 'y^3', 'z', 'z^2', 'z^3', 'x*y', 'x*z', 'y*z', 'x^2y', 'xy^2', 'x^2z', 'xz^2', 'y^2z', 'yz^2', 'xyz']
        self.library_description = poly_desc
        lib_size = len(self.library_description)

        self.vel_field = nn.Linear(lib_size, 3, bias=False)
        self.weight_ref = self.vel_field.weight
        self.target = torch.zeros_like(self.weight_ref, device=device)
    def prune(self, pruning, importance = None):
        torch.nn.utils.prune.l1_unstructured(self.vel_field, 'weight', pruning, importance_scores = importance)
        #self.weight_ref = self.vel_field.weight
    def forward(self, inp):
        x, y, z, t = torch.chunk(inp, 4, dim=1)
        x = x.view(len(x), 1)
        y = y.view(len(y), 1)
        z = z.view(len(z), 1)
        t = t.view(len(t), 1)
        inp = torch.cat([x, y, z, t], dim=1)
        for i, layer in enumerate(self.layers):
            inp = layer(inp)
            if i != len(self.layers) - 1:
                inp = self.actv(inp)
        u = inp
        dt = torch.autograd.grad(u, t, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        adv_out = self.forward_advection(u, x, y, z)
        return u, dt, adv_out

    def forward_advection(self, u, x,
                          y, z):  # computes negative divergence of scalar (in this case, density) * velocity field (defined from van der pol ODE)
        poly = torch.cat(
            [torch.ones_like(x, device=x.device), x, x ** 2, x ** 3, y, y ** 2, y ** 3,
             z, z**2, z**3, x * y, x*z, y*z, x ** 2 * y, x * y ** 2, x**2*z, x*z**2, y**2*z, y*z**2, x*y*z],
            dim=1)
        vel_field = self.vel_field(poly)
        vel_x, vel_y, vel_z = torch.chunk(vel_field, 3, dim=1)
        div_dx = torch.autograd.grad(u * vel_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        div_dy = torch.autograd.grad(u * vel_y, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        div_dz = torch.autograd.grad(u * vel_z, z, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        return -(div_dx + div_dy + div_dz)

    def training_step(self, batch, batch_idx):
        in_xt, labels_u = batch
        in_xt.requires_grad = True
        u, dt, adv_out = self(in_xt.float())
        loss_data = self.mse_loss(u, labels_u.view(len(labels_u), 1).float())
        loss_phys = torch.zeros_like(loss_data, device=loss_data.device)
        if self.current_epoch > self.schedule_reg_epoch:  # start training advection coefficients
            loss_phys = loss_phys + self.l1_loss(self.vel_field.weight, self.target) * self.lam + self.mse_loss(dt,
                                                                                                          adv_out) * self.alpha
        loss = loss_data + loss_phys
        return {"loss": loss, 'loss_data': loss_data, "loss_phys": loss_phys}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        loss_data = torch.stack([x["loss_data"] for x in training_step_outputs]).mean()
        loss_phys = torch.stack([x["loss_phys"] for x in training_step_outputs]).mean()
        self.log('train_loss', loss)
        self.log('train_loss_data', loss_data)
        self.log('train_loss_phys', loss_phys)
        print(self.vel_field.weight)
        print(self.library_description)

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        in_xt, labels_u = batch
        in_xt.requires_grad = True
        u, dt, adv_out = self(in_xt.float())
        loss_data = self.mse_loss(u, labels_u.view(len(labels_u), 1).float())
        loss_phys = torch.zeros_like(loss_data, device=loss_data.device)
        if self.current_epoch > self.schedule_reg_epoch:  # start training advection coefficients
            loss_phys = loss_phys + self.l1_loss(self.vel_field.weight, self.target) * self.lam + self.mse_loss(dt,
                                                                                                          adv_out) * self.alpha
        loss = loss_data + loss_phys
        return {"loss": loss, 'loss_data': loss_data, "loss_phys": loss_phys}

    def validation_epoch_end(self, val_step_outputs):
        loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        loss_data = torch.stack([x["loss_data"] for x in val_step_outputs]).mean()
        loss_phys = torch.stack([x["loss_phys"] for x in val_step_outputs]).mean()
        self.log('val_loss', loss)
        self.log('val_loss_data', loss_data)
        self.log('val_loss_phys', loss_phys)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class NetDiff2D(pl.LightningModule):
    def __init__(self, actv, num_inputs, hidden_units, num_outputs, lr=1e-3, lam=1e-4, alpha=1, device='cpu',
                 schedule_reg_epoch=0):
        super(NetDiff2D, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.schedule_reg_epoch = schedule_reg_epoch
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.lr = lr
        self.lam = lam
        self.device_name = device

        # Assign activation function (exec allows us to assign function from string)
        exec('self.actv = %s' % actv)

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Loop over layers and create each one
        for i in range(len(hidden_units)):
            next_num_inputs = hidden_units[i]
            self.layers += [nn.Linear(num_inputs, next_num_inputs)]
            num_inputs = next_num_inputs

        # Create final layer
        self.layers += [nn.Linear(num_inputs, num_outputs)]

        poly_desc = ['', 'x', 'x^2', 'x^3', 'y', 'y^2', 'y^3', 'x*y', 'x^2y', 'xy^2']
        #derivs_desc = ['', 'u_x', 'u_y']
        #u_desc = ['', 'u', 'u^2']
        derivs_desc=['', 'u_x', 'u_xx', 'u_y', 'u_yy','u_xy']
        u_desc = ['', 'u', 'u^2', 'u^3']
        desc = [a + b + c for (a, b, c) in product(poly_desc, derivs_desc, u_desc)]
        desc[0] = '1'
        self.library_description = desc
        lib_size = len(self.library_description)
        self.regression_layer = nn.Linear(lib_size, num_outputs, bias=False)

        self.weight_ref = self.regression_layer.weight
        self.weight_ref.data.fill_(0.0)  # want to initialize at 0's for the regression coefficients
        self.target = torch.zeros_like(self.weight_ref, device=device)

    def forward(self, inp):
        x, y, t = torch.chunk(inp, 3, dim=1)
        x = x.view(len(x), 1)
        y = y.view(len(y), 1)
        t = t.view(len(t), 1)
        inp = torch.cat([x, y, t], dim=1)
        for i, layer in enumerate(self.layers):
            inp = layer(inp)
            if i != len(self.layers) - 1:
                inp = self.actv(inp)
        u = inp
        dt = torch.autograd.grad(u, t, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dx = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        ddx = torch.autograd.grad(dx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dy = torch.autograd.grad(u, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        ddy = torch.autograd.grad(dy, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dxdy = torch.autograd.grad(dx, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        reg_out = self.forward_regression(u, x, y, dx, ddx, dy, ddy, dxdy)
        return u, dt, reg_out

    # def on_after_backward(self):
    #     all_grads = torch.abs(self.weight_ref.grad)
    #     # all_grads_sorted, _ = torch.sort(all_grads, descending = True, dim = -1)
    #     # self.regression_layer.weight.grad[all_grads < all_grads_sorted[:, 5]] = 0

    def forward_u(self, inp):
        x, t = torch.chunk(inp, 2, dim=1)
        x = x.view(len(x), 1)
        t = t.view(len(t), 1)
        inp = torch.cat([x, t], dim=1)
        for i, layer in enumerate(self.layers):
            inp = layer(inp)
            if i != len(self.layers) - 1:
                inp = self.actv(inp)
        u = inp
        return u

    def forward_regression(self, u, x, y, dx, ddx, dy, ddy, dxdy):
        poly = [1, x, x ** 2, x ** 3, y, y ** 2, y ** 3, x * y, x ** 2 * y, x * y ** 2]
        derivs = [1,dx,ddx,dy,ddy,dxdy]
        #derivs = [1, dx, dy]
        u_pow = [u ** i for i in range(4)]
        phi = torch.cat([a * b * c for (a, b, c) in product(poly, derivs, u_pow)], dim=1)
        return self.regression_layer(phi)

    def forward_phi(self, inp):
        x, y, t = torch.chunk(inp, 3, dim=1)
        x = x.view(len(x), 1)
        y = y.view(len(x), 1)
        t = t.view(len(t), 1)
        inp = torch.cat([x, y, t], dim=1)
        for i, layer in enumerate(self.layers):
            inp = layer(inp)
            if i != len(self.layers) - 1:
                inp = self.actv(inp)
        u = inp
        dt = torch.autograd.grad(u, t, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dx = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        ddx = torch.autograd.grad(dx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dy = torch.autograd.grad(u, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        ddy = torch.autograd.grad(dy, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        dxdy = torch.autograd.grad(dx, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        poly = [1, x, x ** 2, x ** 3, y, y ** 2, y ** 3, x * y, x ** 2 * y, x * y ** 2]
        derivs = [1,dx,ddx,dy,ddy,dxdy]
        #derivs = [1, dx, dy]
        u_pow = [u ** i for i in range(4)]
        phi = torch.cat([a * b * c for (a, b, c) in product(poly, derivs, u_pow)], dim=1)
        return dt, phi

    def training_step(self, batch, batch_idx):
        in_xt, labels_u = batch
        in_xt.requires_grad = True
        u, dt, reg_out = self(in_xt.float())
        loss_data = self.mse_loss(u, labels_u.view(len(labels_u), 1).float())
        loss_phys = torch.zeros_like(loss_data, device=loss_data.device)
        if self.current_epoch > self.schedule_reg_epoch:  # start training regression coefficients
            loss_phys = loss_phys + self.l1_loss(self.weight_ref, self.target) * self.lam + self.mse_loss(dt,
                                                                                                          reg_out) * self.alpha
        loss = loss_data + loss_phys
        return {"loss": loss, 'loss_data': loss_data, "loss_phys": loss_phys}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        loss_data = torch.stack([x["loss_data"] for x in training_step_outputs]).mean()
        loss_phys = torch.stack([x["loss_phys"] for x in training_step_outputs]).mean()
        self.log('train_loss', loss)
        self.log('train_loss_data', loss_data)
        self.log('train_loss_phys', loss_phys)

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        in_xt, labels_u = batch
        in_xt.requires_grad = True
        u, dt, reg_out = self(in_xt.float())
        loss_data = self.mse_loss(u, labels_u.view(len(labels_u), 1).float())
        if self.current_epoch <= self.schedule_reg_epoch:
            dt = dt.detach()
        loss_phys = self.l1_loss(self.weight_ref, self.target) * self.lam + self.mse_loss(dt,reg_out) * self.alpha
        loss = loss_data + loss_phys
        return {"loss": loss, 'loss_data': loss_data, "loss_phys": loss_phys}

    def validation_epoch_end(self, val_step_outputs):
        loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        loss_data = torch.stack([x["loss_data"] for x in val_step_outputs]).mean()
        loss_phys = torch.stack([x["loss_phys"] for x in val_step_outputs]).mean()
        self.log('val_loss', loss)
        self.log('val_loss_data', loss_data)
        self.log('val_loss_phys', loss_phys)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class NetDiff2DAdvection(pl.LightningModule):
    def __init__(self, actv, num_inputs, hidden_units, num_outputs, lr=1e-3, lam=1e-4, alpha=1, device='cpu',
                 schedule_reg_epoch=0):
        super(NetDiff2DAdvection, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.schedule_reg_epoch = schedule_reg_epoch
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.lr = lr
        self.lam = lam
        self.device_name = device

        # Assign activation function (exec allows us to assign function from string)
        exec('self.actv = %s' % actv)

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        # Loop over layers and create each one
        for i in range(len(hidden_units)):
            next_num_inputs = hidden_units[i]
            self.layers += [nn.Linear(num_inputs, next_num_inputs)]
            num_inputs = next_num_inputs

        # Create final layer
        self.layers += [nn.Linear(num_inputs, num_outputs)]

        poly_desc = ['1', 'x', 'x^2', 'x^3', 'y', 'y^2', 'y^3', 'x*y', 'x^2y', 'xy^2']
        self.library_description = poly_desc
        lib_size = len(self.library_description)

        self.vel_field = nn.Linear(lib_size, 2, bias=False)
        self.weight_ref = self.vel_field.weight
        self.target = torch.zeros_like(self.weight_ref, device=device)
    def prune(self, pruning, importance = None):
        torch.nn.utils.prune.l1_unstructured(self.vel_field, 'weight', pruning, importance_scores = importance)
        #self.weight_ref = self.vel_field.weight
    def forward(self, inp):
        x, y, t = torch.chunk(inp, 3, dim=1)
        x = x.view(len(x), 1)
        y = y.view(len(y), 1)
        t = t.view(len(t), 1)
        inp = torch.cat([x, y, t], dim=1)
        for i, layer in enumerate(self.layers):
            inp = layer(inp)
            if i != len(self.layers) - 1:
                inp = self.actv(inp)
        u = inp
        dt = torch.autograd.grad(u, t, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        adv_out = self.forward_advection(u, x, y)
        return u, dt, adv_out

    def forward_u(self, inp):
        x, t = torch.chunk(inp, 2, dim=1)
        x = x.view(len(x), 1)
        t = t.view(len(t), 1)
        inp = torch.cat([x, t], dim=1)
        for i, layer in enumerate(self.layers):
            inp = layer(inp)
            if i != len(self.layers) - 1:
                inp = self.actv(inp)
        u = inp
        return u

    def forward_advection(self, u, x,
                          y):  # computes negative divergence of scalar (in this case, density) * velocity field (defined from van der pol ODE)
        poly = torch.cat(
            [torch.ones_like(x, device=x.device), x, x ** 2, x ** 3, y, y ** 2, y ** 3, x * y, x ** 2 * y, x * y ** 2],
            dim=1)
        vel_field = self.vel_field(poly)
        vel_x, vel_y = torch.chunk(vel_field, 2, dim=1)
        div_dx = \
        torch.autograd.grad(u * vel_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        div_dy = \
        torch.autograd.grad(u * vel_y, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        return -(div_dx + div_dy)

    def training_step(self, batch, batch_idx):
        in_xt, labels_u = batch
        in_xt.requires_grad = True
        u, dt, adv_out = self(in_xt.float())
        if self.current_epoch <= self.schedule_reg_epoch:
            dt = dt.detach()
        loss_data = self.mse_loss(u, labels_u.view(len(labels_u), 1).float()) # start training advection coefficients
        loss_phys = self.l1_loss(self.vel_field.weight, self.target) * self.lam + self.mse_loss(dt,adv_out) * self.alpha
        loss = loss_data + loss_phys
        return {"loss": loss, 'loss_data': loss_data, "loss_phys": loss_phys}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        loss_data = torch.stack([x["loss_data"] for x in training_step_outputs]).mean()
        loss_phys = torch.stack([x["loss_phys"] for x in training_step_outputs]).mean()
        self.log('train_loss', loss)
        self.log('train_loss_data', loss_data)
        self.log('train_loss_phys', loss_phys)
        if self.current_epoch % 10 == 0:
            print(self.vel_field.weight)
            print(self.library_description)

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        in_xt, labels_u = batch
        in_xt.requires_grad = True
        u, dt, adv_out = self(in_xt.float())
        loss_data = self.mse_loss(u, labels_u.view(len(labels_u), 1).float())
        loss_phys = torch.zeros_like(loss_data, device=loss_data.device)
        if self.current_epoch > self.schedule_reg_epoch:  # start training advection coefficients
            loss_phys = loss_phys + self.l1_loss(self.vel_field.weight, self.target) * self.lam + self.mse_loss(dt,
                                                                                                          adv_out) * self.alpha
        loss = loss_data + loss_phys
        return {"loss": loss, 'loss_data': loss_data, "loss_phys": loss_phys}

    def validation_epoch_end(self, val_step_outputs):
        loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        loss_data = torch.stack([x["loss_data"] for x in val_step_outputs]).mean()
        loss_phys = torch.stack([x["loss_phys"] for x in val_step_outputs]).mean()
        self.log('val_loss', loss)
        self.log('val_loss_data', loss_data)
        self.log('val_loss_phys', loss_phys)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def get_all_spectra(net, loader, n_output, device, n_reg = 16):
  F_u = FIM(model = net, loader = loader, representation = PMatDense, n_output = n_output, variant = 'regression', function = lambda *x: net(x[0].float())[0], device = device)
  F_u.compute_eigendecomposition()

  F_ut = FIM(model = net, loader = loader, representation = PMatDense, n_output = n_output, variant = 'regression', function = lambda *x: net(x[0].float())[1], device = device)
  F_ut.compute_eigendecomposition()

  F_reg = FIM(model = net, loader = loader, representation = PMatDense, n_output = n_output, variant = 'regression', function = lambda *x: net(x[0].float())[2], device = device)
  F_reg.compute_eigendecomposition()

  F_ud, F_utd, F_regd = F_u.get_dense_tensor().detach(), F_ut.get_dense_tensor().detach(), F_reg.get_dense_tensor().detach()
  reg_FIMs = [F_ud[-n_reg:, -n_reg:], F_utd[-n_reg:, -n_reg:], F_regd[-n_reg:, -n_reg:]]
  cond_numbers = [torch.linalg.cond(F_ud), torch.linalg.cond(F_utd), torch.linalg.cond(F_regd)]
  return [F_u.evals, F_ut.evals, F_reg.evals], [F_u.evecs, F_ut.evecs, F_reg.evecs], cond_numbers, reg_FIMs

def prune_iteratively(smoothing = 3, base_level = 3, epochs_sched = [100, 40, 40, 40], pruning_sched = [0,4,4,4], use_FIM = False, preload = None):
    n_levels = len(epochs_sched)
    print('beginning')
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    num_units = 40
    num_layers = 6
    #synth_args = {'x_range': [-6, 6], 'y_range': [-6, 6], 'num_x': 256, 'num_y': 256, 't_max': 10, 'dt': 0.05, 'mu': 2,
    #              'n': 300000, 'r': 3, 'bandwidth': 0.0, 'ic_mode': 'circle'}
    synth_args = {'x_range': [-3, 3], 'y_range': [-3, 3], 'z_range': [0, 5], 'num_x': 100, 'num_y': 100, 'num_z': 100,
                  't_max': 10, 'dt': 0.01, 'n': 300000, 'spherical_r': 2.0}
    already_generated = pk.load(open("already_generated_300k_lorenz_spherical.pk", "rb"))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    seed_all(1729)
    # full_data = DataVDP(synth_args, subsample_zero=0.05, subsample_all=1, smoothing=smoothing,
    #                     already_generated=already_generated, subsample_base=base_level)
    full_data = DataLorenz(synth_args, subsample_zero=1e-4, subsample_all=0.01, smoothing=smoothing,
                           already_generated=already_generated, subsample_base=base_level)
    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size
    generator = torch.Generator()
    generator.manual_seed(1729)
    train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size],
                                                                generator=generator)
    seed_all(1729)
    train_loader = DataLoader(train_dataset, batch_size=512, num_workers=4, shuffle=True)

    val_loader = DataLoader(test_dataset, batch_size=512, num_workers=4)

    # net = NetDiff2DAdvection(actv='nn.Tanh()',
    #                          num_inputs=3,
    #                          hidden_units=[num_units] * num_layers,
    #                          num_outputs=1, lam=1e-6, device=device, alpha=0.2, lr=1e-3,
    #                          schedule_reg_epoch=0)
    net = NetDiff3DAdvection(actv='nn.Tanh()',
                             num_inputs=4,
                             hidden_units=[num_units] * num_layers,
                             num_outputs=1, lam=1e-8, device=device, alpha=0.1, lr=1e-3,
                             schedule_reg_epoch=0)
    net = net.to(device)
    if preload is not None:
        net.load_state_dict(torch.load(preload))

    for i in range(n_levels):
        print("Pruning level", i)
        if i == 0:
            net.schedule_reg_epoch = epochs_sched[i] // 2
            print(net.vel_field.weight)
            print(net.library_description)
            if preload is None:
                trainer = pl.Trainer(gpus=1 if use_cuda else 0, max_epochs=epochs_sched[i], logger=tb_logger,
                                     progress_bar_refresh_rate=100)
                trainer.fit(net, train_loader, val_dataloaders=None)
        else:
            net.schedule_reg_epoch = 0
            trainer = pl.Trainer(gpus=1 if use_cuda else 0, max_epochs=epochs_sched[i], logger=tb_logger, progress_bar_refresh_rate=100)
            trainer.fit(net, train_loader, val_dataloaders=None)
        print('PRUNING')
        if use_FIM:
            _, _, _, reg_FIMs = get_all_spectra(net, val_loader, 1, device='cpu')
            net.prune(pruning_sched[i], importance = torch.diag(reg_FIMs[2]).reshape(net.vel_field.weight))
        else:
            net.prune(pruning_sched[i], importance = None)
        net_name = 'model_lorenz_spherical_pruning_seeded_' + str(i)
        if use_FIM:
            net_name += '_FIM'
        net_name += '.pth'
        #if i != 0 or preload is None:
        torch.save(net.state_dict(), net_name)
        print(net.vel_field.weight)
        print(net.library_description)

def get_all_FIMs():
    num_units = 40
    num_layers = 6
    all_evals, all_evecs, all_weights = [], [], []
    all_reg_FIMs = []
    synth_args = {'x_range': [-6, 6], 'y_range': [-6, 6], 'num_x': 512, 'num_y': 512, 't_max': 10, 'dt': 0.05, 'mu': 2,
                  'n': 300000, 'r': 3, 'bandwidth': 0.0}

    full_data_0 = DataVDP(synth_args, subsample_zero=0.1, subsample_all=1, smoothing=0)

    already_generated = {'u': full_data_0.u, 'x': full_data_0.x, 'y': full_data_0.y, 't': full_data_0.t}
    pk.dump(already_generated, open("already_generated_300k.pk", "wb"))
    #already_generated = pk.load(open("already_generated_300k.pk", "rb"))

    for i in range(7):
        seed_all(1729)
        full_data = DataVDP(synth_args, subsample_zero=0.05, subsample_all=0.5, smoothing=i,
                            already_generated=already_generated, subsample_base=3)
        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size
        generator = torch.Generator()
        generator.manual_seed(1729)
        train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size],
                                                                    generator=generator)
        seed_all(1729)
        train_loader = DataLoader(train_dataset, batch_size=512, num_workers=4, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=512, num_workers=4)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        net = NetDiff2DAdvection(actv='nn.Tanh()',
                                 num_inputs=3,
                                 hidden_units=[num_units] * num_layers,
                                 num_outputs=1, lam=1e-5, device=device, alpha=0.2, lr=1e-3, schedule_reg_epoch=100)
        net = net.to('cpu')
        net.load_state_dict(torch.load('model_vdp_advection_highres_seeded_' + str(i) + '.pth'))
        evals, evecs, cond_numbers, reg_FIMs = get_all_spectra(net, val_loader, 1, device='cpu', n_reg=20)
        all_reg_FIMs.append(reg_FIMs)
        all_evals.append(evals)
        all_evecs.append(evecs)
        all_weights.append(net.vel_field.weight.detach().numpy())
        print(net.vel_field.weight)
        print(net.library_description)
        print(cond_numbers)
    pk.dump([all_reg_FIMs, all_evals, all_evecs, all_weights], open("vdp_fims_evals_evecs_weights.pk", "wb"))\

def run_vdp():
    #MOVE SUBSAMPLE LOGIC!!!!
    #prune_iteratively(smoothing = 3, base_level=3, epochs_sched = [100,40,40], pruning_sched = [4,4,4],use_FIM = False, preload = 'model_vdp_advection_pruning_seeded_0.pth')
    #get_FIMs()
    print('beginning')
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    #checkpoint_callback = pl_callbacks.ModelCheckpoint(every_n_epochs=)
    num_units = 40
    num_layers = 6
    synth_args = {'x_range': [-6, 6], 'y_range': [-6, 6], 'num_x': 256, 'num_y': 256, 't_max': 10, 'dt': 0.05, 'mu': 2,
                  'n': 300000, 'r': 3, 'bandwidth': 0.0, 'ic_mode': 'circle'}
    #full_data_0 = DataVDP(synth_args, subsample_zero=0.1, subsample_all=1, smoothing=0)
    #already_generated = {'u': full_data_0.u, 'x': full_data_0.x, 'y': full_data_0.y, 't': full_data_0.t}
    #pk.dump(already_generated, open("already_generated_300k_uniform.pk", "wb"))
    already_generated = pk.load(open("already_generated_300k.pk","rb"))
    for i in range(7):
        print("Coarsening Level", i)
        seed_all(1729)
        full_data = DataVDP(synth_args, subsample_zero = 0.01, subsample_all = 0.5, smoothing = i, already_generated = already_generated, subsample_base=3)
        #full_data = DataVDPST(synth_args, smoothing = i, already_generated=already_generated, st_subsample=.03)
        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size
        generator = torch.Generator()
        generator.manual_seed(1729)
        train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size],
                                                                    generator=generator)
        seed_all(1729)
        train_loader = DataLoader(train_dataset, batch_size=512, num_workers=4, shuffle=True)

        val_loader = DataLoader(test_dataset, batch_size=512, num_workers=4)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # net = NetDiff2DAdvection(actv='nn.Tanh()',
        #                          num_inputs=3,
        #                          hidden_units=[num_units] * num_layers,
        #                          num_outputs=1, lam=1e-8, device=device, alpha=1e-2, lr=1e-3, schedule_reg_epoch=50)
        net = NetDiff2D(actv='nn.Tanh()',
                                 num_inputs=3,
                                 hidden_units=[num_units] * num_layers,
                                 num_outputs=1, lam=1e-8, device=device, alpha=1e-2, lr=1e-3, schedule_reg_epoch=50)
        net = net.to(device)

        trainer = pl.Trainer(gpus=1 if use_cuda else 0, max_epochs=80, logger=tb_logger, progress_bar_refresh_rate=100)
        trainer.fit(net, train_loader, val_dataloaders=None)
        torch.save(net.state_dict(), 'model_vdp_420_seeded_' + str(i) + '.pth')
        print(net.weight_ref)  #
        print(net.library_description)

def run_lorenz():
    print('beginning')
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    num_units = 40
    num_layers = 6
    synth_args = {'x_range': [-3, 3], 'y_range': [-3, 3], 'z_range': [0,5], 'num_x': 100, 'num_y': 100, 'num_z': 100, 't_max': 10, 'dt': 0.01, 'n': 300000, 'spherical_r':2.0}
    #full_data_0 = DataLorenz(synth_args, subsample_zero=0.05, subsample_all=0.05, smoothing=0)
    #already_generated = {'u': full_data_0.u, 'x': full_data_0.x, 'y': full_data_0.y, 'z': full_data_0.z, 't': full_data_0.t}
    #pk.dump(already_generated, open("already_generated_300k_lorenz_spherical.pk", "wb"))
    already_generated = pk.load(open("already_generated_300k_lorenz_spherical.pk","rb"))
    for i in range(7):
        print("Coarsening Level", i)
        seed_all(1729)
        #full_data = DataLorenz(synth_args, subsample_zero = 1e-4, subsample_all = 0.1, smoothing = i, already_generated = already_generated, subsample_base=2)
        full_data = DataLorenzST(synth_args, smoothing = i / 2, already_generated=already_generated, st_subsample=3e-3)
        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size
        generator = torch.Generator()
        generator.manual_seed(1729)
        train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size],
                                                                    generator=generator)
        seed_all(1729)
        train_loader = DataLoader(train_dataset, batch_size=1001, num_workers=4, shuffle=False)

        val_loader = DataLoader(test_dataset, batch_size=1001, num_workers=4)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        net = NetDiff3DAdvection(actv='nn.Tanh()',
                                 num_inputs=4,
                                 hidden_units=[num_units] * num_layers,
                                 num_outputs=1, lam=0, device=device, alpha=1e-2, lr=1e-3, schedule_reg_epoch=50)
        net = net.to(device)
        importance_answers = torch.zeros_like(net.vel_field.weight)
        print(importance_answers.shape)
        importance_answers[0,[1,4]] = 1.0
        importance_answers[1,[1,4,11]] = 1.0
        importance_answers[2,[7,10]] = 1.0
        net.prune(53, importance = importance_answers)
        print(net.vel_field.weight)
        trainer = pl.Trainer(gpus=1 if use_cuda else 0, max_epochs=80, logger=tb_logger, progress_bar_refresh_rate=1000)
        trainer.fit(net, train_loader, val_dataloaders=None)
        torch.save(net.state_dict(), 'model_lorenz_spherical_advection_prepruned_seeded_' + str(i) + '.pth')
        print(net.weight_ref)  #
        print(net.library_description)
if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system')
    #prune_iteratively(smoothing = 0, base_level=3, epochs_sched = [100, 40, 40, 40, 40, 40], pruning_sched = [8,8,8,8,8,8],use_FIM = False, preload = 'model_lorenz_spherical_advection_seeded_0.pth')
    #get_FIMs()
    #run_vdp()
    #run_lorenz()
    net = NetDiff2D(actv='nn.Tanh()',
                    num_inputs=3,
                    hidden_units=[40] * 6,
                    num_outputs=1, lam=1e-8, device='cpu', alpha=1e-2, lr=1e-3, schedule_reg_epoch=50)
    lib = net.library_description
    print(lib.index('u'))
    print(lib.index('x^2u'))
    print(lib.index('yu_x'))
    print(lib.index('yu_y'))
    print(lib.index('xu_y'))
    print(lib.index('x^2yu_y'))
    print(len(lib))