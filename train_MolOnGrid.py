from options import read_options_yaml, print_options_yaml, define_train, compare_opts
from IO import save_check, load_check
from molecule import Molecule
from grids import SamplingCoords
from data import MolOnGrid, collate_fn_MolOnGrid
import torch
import numpy as np
import pickle
import os.path as osp
from torch.utils.data import DataLoader
import gen_molecules


opts = read_options_yaml('./options.yaml')
device = 'cuda'
guide_rate = opts['train']['guide_rate']

print("==============OPTIONS==============")
print_options_yaml(opts)

# define model and optimzer
tr_opts = opts['train']
model, optimizer, loss_func = define_train(tr_opts)
ini_itr, max_itr = 0, tr_opts['max_itr']
if 'restart_from' in tr_opts and tr_opts['restart_from'] is not None:
    restart_path = tr_opts['restart_from']
    if 'redefine_from_restart' in tr_opts:
        if 'model' in tr_opts['redefine_from_restart']: model = None
        if 'optimizer' in tr_opts['redefine_from_restart']: optimizer = None
    ini_itr, model, optimizer, ini_loss = load_check(restart_path, model=model, optimizer=optimizer)


# prepare all structures
dataset = MolOnGrid(opts['molecules'], opts['sampling'], path='./', device=device)
dataset.gen_all()
for i_struc, data in enumerate(dataset):
    with open('./results/coords%d.pkl'%(i_struc), 'wb') as f_c:
        pickle.dump(data['mol']._grid.coords, f_c)

# training
batch_size = 3
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_MolOnGrid)
model = model.to(device)
max_itr, max_iks = tr_opts['max_itr'], tr_opts['max_iks']
all_rho_diff = []
all_v = []
itr_save, iks_save = tr_opts['itr_save_every'], tr_opts['iks_save_every']
for itr in range(ini_itr, ini_itr+max_itr):
    for i_batch, data in enumerate(train_loader):
        mol_batch, x_batch, grid_sizes = data.values()
        # through model
        x_batch = x_batch.to(device)
        for iks in range(max_iks):
            optimizer.zero_grad()
            y_batch = model(x_batch)
            grid_bg = 0
            batch_loss = 0.0
            # structures in the batch
            for i_struc_in_batch, mol in enumerate(mol_batch):
                grid_range = range(grid_bg, grid_bg + grid_sizes[i_struc_in_batch])
                y = y_batch[grid_range].mean(1)
                vxc = y.detach().to('cpu').numpy()
                # calculate new density
                vxc_mat = mol.V_mat(vxc, package='numpy')
                dm_new = mol.dm_from_Vxc_mat(vxc_mat)  # KS
                rho_new = mol.density_on_grid(dm_new, deriv=0, package='torch', device=device).squeeze(0).float()
                # use new density for loss
                rho_diff = rho_new - mol.rho('ccsd', package='torch', device=device).squeeze(0).float()
                mean_rho_diff, max_rho_diff = rho_diff.abs().mean().item(), rho_diff.abs().max().item()
                i_struc = i_batch*batch_size + i_struc_in_batch
                print("itr %d,istruc %d,iks %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(itr, i_struc, iks, mean_rho_diff, max_rho_diff))
                target = y.detach() + guide_rate * rho_diff
                target -= target.mean()
                batch_loss += loss_func(y, target)
                grid_bg += grid_sizes[i_struc_in_batch]
                # save
                if itr % itr_save == 0 and iks % iks_save == 0:
                    all_rho_diff += [rho_diff]
                    all_v += [vxc]
                    with open('./results/rho_diff%d.pkl'%(i_struc), 'wb') as f_rho:
                        pickle.dump(all_rho_diff.cpu().numpy(), f_rho)
                    with open('./results/vxc%d.pkl'%(i_struc), 'wb') as f_v:
                        pickle.dump(all_v, f_v)
                    save_check('./checkpoints', itr, model, optimizer, batch_loss)
                    print("itr: %d\ti_struc: %d\tiks: %d\tloss: %10.3e"%(itr, i_struc, iks, batch_loss.item()))
            # backpropagation
            batch_loss.backward()
            optimizer.step()
            