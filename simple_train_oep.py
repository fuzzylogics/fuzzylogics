from options import read_options_yaml, print_options_yaml, define_train, compare_opts
from IO import save_check, load_check
from molecule import Molecule
from grids import SamplingCoords
from data import MolOnGrid
import torch
import numpy as np
import pickle
import os.path as osp
from torch.utils.data import DataLoader
import gen_molecules


opts = read_options_yaml('./options.yaml')
device = 'cuda'

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
loading_data = False
if 'load_path' in opts['data'] and opts['data']['load_path'] is not None:
    with open(opts['data']['load_path'], 'rb') as f:
        saved_dict = pickle.load(f)
    xs, mols, saved_opts = saved_dict['xs'], saved_dict['mols'], saved_dict['opts']
    if compare_opts(opts['molecules'], saved_opts['molecules']) and compare_opts(opts['sampling'], saved_opts['sampling']):
        loading_data = True
if not loading_data:
    print('options are different this time, overwritting data ...')
    assert opts['molecules']['struc_from'] == 'function'
    gen_mols_func = eval('gen_molecules.' + opts['molecules']['struc_func']['name'])
    mols = gen_mols_func(**opts['molecules']['struc_func']['arguments'])
    n_struc = len(mols)
    xs = []
    ts = []
    for i_struc, mol in enumerate(mols):
        print("generating necessary data for mol: %d."%(i_struc))
        mol.generate_all(opts['molecules']['grids']['grid_level'])
        sampling_coords = SamplingCoords(mol.grid, sampling_opts=opts['sampling'])
        sampling_coords.mesh_out()
        rho, _ = mol.rho_n_save('ccsd', grid=sampling_coords, deriv=(0,1), package='torch', device=device)
        # HF
        rho_hf = mol.density_on_grid(mol.dm_hf)
        rho_diff = rho_hf - mol.rho('ccsd')
        mean_rho_diff, max_rho_diff = np.mean(np.abs(rho_diff)), np.max(np.abs(rho_diff))
        print("i_struc = %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(i_struc, mean_rho_diff, max_rho_diff))

        # oep 
        # rho_oep = mol.density_on_grid(mol.dm_oep)
        # rho_diff = rho_oep - mol.rho('ccsd')
        # mean_rho_diff, max_rho_diff = np.mean(np.abs(rho_diff)), np.max(np.abs(rho_diff))
        # print("i_struc = %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(i_struc, mean_rho_diff, max_rho_diff))

        # np.save('./dm_oep.npy', mol.dm_oep)
        # np.save('./dm_ccsd.npy', mol.dm_ccsd)
        # mol.write_structure('./H2O.str')

        xs += [rho]

        vxc = mol.oep_on_grid()
        ts += [vxc]

        del(mol._pyscf_mol)

    with open('./data_rho/data.pkl', 'wb') as f:
        pickle.dump({'xs':xs,'mols':mols}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
for i_struc, mol in enumerate(mols):
    with open('./results/coords%d.pkl'%(i_struc), 'wb') as f_c:
        pickle.dump(mol._grid.coords, f_c)


for i_struc in range(len(xs)):
    xs[i_struc] = torch.from_numpy(xs[i_struc]).float().transpose(1,0).squeeze().contiguous()
    print('xs', xs[i_struc].shape)

model = model.to(device)
max_itr, max_iks = tr_opts['max_itr'], tr_opts['max_iks']
all_rho_diff = []
all_v = []
itr_save, iks_save = tr_opts['itr_save_every'], tr_opts['iks_save_every']
for itr in range(ini_itr, ini_itr+max_itr):
    for i_struc, mol in enumerate(mols):
        x, target = xs[i_struc], ts[i_struc]
        # through model
        x = x.to(device)
        for iks in range(max_iks):
            optimizer.zero_grad()
            y = model(x)

            vxc = y.detach().mean(1).to('cpu').numpy()
            # # calculate new density
            # vxc_mat = mol.V_mat(vxc, package='numpy')
            # dm_new = mol.dm_from_Vxc_mat(vxc_mat)
            # rho_new = mol.density_on_grid(dm_new)
            # # evaluate new density
            # rho_diff = rho_new - mol.rho('ccsd')
            # mean_rho_diff, max_rho_diff = np.mean(np.abs(rho_diff)), np.max(np.abs(rho_diff))
            # print("itr %d,istruc %d,iks %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(itr, iks, i_struc, mean_rho_diff, max_rho_diff))
            # target = y.detach().squeeze() + guide_rate * torch.from_numpy(rho_diff).float().to(device).squeeze()
            

            loss = loss_func(y.squeeze(), target)
            loss.backward()
            optimizer.step()

            if itr % itr_save == 0 and iks % iks_save == 0:
                all_rho_diff += [rho_diff]
                all_v += [vxc]
                with open('./results/rho_diff%d.pkl'%(i_struc), 'wb') as f_rho:
                    pickle.dump(all_rho_diff, f_rho)
                with open('./results/vxc%d.pkl'%(i_struc), 'wb') as f_v:
                    pickle.dump(all_v, f_v)
                save_check('./checkpoints', itr, model, optimizer, loss)
                print("itr: %d\ti_struc: %d\tiks: %d\tloss: %10.3e"%(itr, i_struc, iks, loss.item()))
            