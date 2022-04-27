from nngeometry.generator import Jacobian
from nngeometry.object import FMatDense, PMatDense
from nngeometry.metrics import FIM
import pickle as pk
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from NNetModel import DataFixed, Net
from torch.utils.data import Dataset, DataLoader, IterableDataset

if __name__ == '__main__':
    net = Net(actv='ReLU()',
                  num_inputs=5,
                  hidden_units=[20, 20, 20, 20, 20, 20, 20, 20],
                  num_outputs=1)
    net.load_state_dict(torch.load('model_sec_half.pth'))
    net.eval()

    full_data = DataFixed(perturbation='', coarsening_level=1)
    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers = 4, shuffle = True)

    val_loader = DataLoader(test_dataset, batch_size=64, num_workers = 4)


    F = FIM(model = net, loader = val_loader, representation=PMatDense, n_output = 1, variant = 'regression', device = 'cpu')
    F.compute_eigendecomposition()
    print(F.evals, F.evecs)
    pk.dump([F.data.detach().numpy(), F.evals.detach().numpy(), F.evecs.detach().numpy()],open('FIM_evals_evecs_sec_half.pk',"wb"))


