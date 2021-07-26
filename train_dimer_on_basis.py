# train dimer on basis

import sys
import numpy as np
import torch
import models
import pyscf
from torch.utils.tensorboard import SummaryWriter
import time

from molecule import Molecule
from options import read_options_yaml, define_train
from IO import save_check, load_check
from torch.utils.data import DataLoader
from data import MolOnBasis4Graph_InMem, collate_fn_graph


# loading data and config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)
opts = read_options_yaml(sys.argv[1])
tb_writer = SummaryWriter('./tb')

# molecules
mol_opts = opts['molecules']
if 'struc_from' not in opts['molecules']:
    mol_opts['struc_from'] = 'dimer_range'
else: mol_opts['struc_from'] = 'dimer_range'


# define model and optimzer
tr_opts = opts['train']
model, optimizer, loss_func = define_train(tr_opts)
ini_epoch, max_epochs = 0, tr_opts['max_epoch']
if 'restart_from' in tr_opts and tr_opts['restart_from'] is not None:
    restart_path = tr_opts['restart_from']
    if 'redefine_from_restart' in tr_opts:
        if 'model' in tr_opts['redefine_from_restart']: model = None
        if 'optimizer' in tr_opts['redefine_from_restart']: optimizer = None
    ini_epoch, model, optimizer, ini_loss = load_check(restart_path, model=model, optimizer=optimizer)

# prepare oep
if 'load_from' in opts['data'] and opts['data']['load_from'] is not None:
    import pickle
    with open(opts['data']['load_from'], 'rb') as f:
        dataset = pickle.load(f)
    dataset.to_device(device)
else:
    dataset = MolOnBasis4Graph_InMem(opts)
    data_prepare = DataLoader(dataset)
    for i, data in enumerate(data_prepare):
        print('Finished molecule #: ', i)
        # print(data['x'].shape, data['e'].shape, data['target'].shape)
    dataset.normalize_together()
    dataset.save_data('./')

# normalize data
if not dataset.normalized:
    dataset.normalize_together()

for i, data in enumerate(dataset):
    print(data['x'].shape, data['e'].shape, data['target'].shape)
    # print(data['x'], data['e'], data['target'])

# training
train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_graph)
eval_loader = DataLoader(dataset)
t0 = time.time()
t2 = t0
for epoch in range(ini_epoch, ini_epoch+max_epochs):
    optimizer.zero_grad()
    for data in train_loader:
        inputs = [data['x'], data['edge_index'], data['e']]
        target = data['target']
        output = model(*inputs)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
    if epoch % opts['train']['eval_every_x_epoch'] == 0:
        t1 = time.time()
        total_loss = 0
        for data in eval_loader:
            loss = loss_func(output, target)
            total_loss += loss.item()
        t2o, t2 = t2, time.time()
        print("epoch: %d\tloss: %10.3e\ttime: (train: %3.3f, eval: %3.3f)"%(epoch, total_loss, t1-t2o, t2-t1))
        
        tb_writer.add_scalar('Loss/train', total_loss, epoch)

output = model(*inputs)
tb_writer.flush()
save_check(ini_epoch+max_epochs, model, optimizer, loss)















# analyze results
# loss = loss_func(output, target)
# print("epoch = %d\tloss = %10.3e" % (ini_epoch+max_epochs, loss.item()))
# output = output.detach().cpu().numpy().reshape(-1)
# np.save("./output%d.npy" %(ini_epoch+max_epochs), output)





        # print("max_output = ", output.max().item(), "min_output = ", output.min().item())
        # print("max_target = ", target.max().item(), "mean_target = , ", target.mean().item(), "min_target = ", target.min().item())
        # print("epoch = %d\tloss = %10.3e" % (epoch, loss.item()))

# # print(output, target)

# vxc_coeff_ml = output
# vxc_mat_ml = np.einsum('ijb,b->ij', integral_3c1e, vxc_coeff_ml)
# _, _, dm_ml = ks.solve_KS(H+vxc_mat_ml, S, mo_occ)
# rho_ml = ks.density_on_grids(h2.mol, coords, dm_ml, deriv=0)

# err_hf = ks.density_on_grids(h2.mol, coords, dm_hf-h2.dm_ccsd, deriv=0)
# err_oep = ks.density_on_grids(h2.mol, coords, dm-h2.dm_ccsd, deriv=0)
# err_ml = ks.density_on_grids(h2.mol, coords, dm_ml-h2.dm_ccsd, deriv=0)
# print(err_ml)
# print("error max = %10.3e; error mean = %10.3e" % (np.abs(err_ml).max(), np.abs(err_ml).mean()) )
# np.savez("./rho_error%d.npz" %(ini_epoch+max_epochs), coords=coords, hf=err_hf, oep=err_oep, ml=err_ml)

