
# from torch.utils.data.dataset import T
from options import read_options_yaml, print_options_yaml, define_test, gen_test_opts
from IO import save_check, load_check
from molecule import Molecule
from grids import SamplingCoords
from data import MolOnGrid, collate_fn_MolOnGrid, eval_I
import torch
import numpy as np
import pickle
import os
import os.path as osp
import shutil
from torch.utils.data import DataLoader
import gen_molecules



opts = read_options_yaml('./options.yaml')
device = 'cuda'
print("==============OPTIONS==============")
print_options_yaml(opts)
test_opts = gen_test_opts(opts)
print_options_yaml(test_opts)

# define model and optimzer
if test_opts['checkpoint']['from_DDP']:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=0, world_size=1)
model = define_test(test_opts, is_ddp=test_opts['checkpoint']['from_DDP'], device=device)
assert 'checkpoint' in test_opts and test_opts['checkpoint']['path'] is not None
checkpoint_path = test_opts['checkpoint']['path']
if 'state_dict_only' in test_opts['checkpoint'] and test_opts['checkpoint']['state_dict_only']:
    model.load_state_dict(torch.load(checkpoint_path))
else:
    itr, model, _, loss = load_check(checkpoint_path, model=model, optimizer=None)

# test all structures
path = './test1'
shutil.copyfile(checkpoint_path, osp.join(path, 'checkpoint.chk'))
dataset = MolOnGrid(test_opts['molecules'], test_opts['sampling'], path=path, device=device)
test_err = np.zeros((6, len(dataset)))
for i_struc, data in enumerate(dataset):
    mol, x = data['mol'], data['x']
    y = model(x.to(device)).mean(1)
    vxc = y.detach()
    # calculate new density
    vxc_mat = mol.V_mat(vxc, package='torch', device=device)
    dm_new = mol.dm_from_Vxc_mat(vxc_mat)  # KS
    rho_new = mol.density_on_grid(dm_new, deriv=0, package='torch', device=device).squeeze(0).float()
    # use new density for loss
    rho_target = mol.rho('ccsd', package='torch', device=device).squeeze(0).float()
    rho_diff = rho_new - rho_target
    mean_rho_diff, max_rho_diff = rho_diff.abs().mean().item(), rho_diff.abs().max().item()
    print("i_struc %d,\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(i_struc, mean_rho_diff, max_rho_diff))
    
    err_I = eval_I(rho_new, rho_target, mol.grid.weights)
    print("errorItg = %10.8e"%(err_I))

    # # scf
    # sampling_coords = SamplingCoords(mol.grid, test_opts['sampling'])
    # sampling_coords.mesh_out()
    # dm_scf = mol.dm_scf_ml(model, sampling_coords, mol.dm_dft, device=device, input_shape='vanilla')
    # dm_scf_diff = dm_scf-mol.dm_ccsd
    # rho_scf_diff = mol.density_on_grid(dm_scf_diff, deriv=0, package='torch', device=device).squeeze(0).float()
    # rho_scf = mol.density_on_grid(dm_scf, deriv=0, package='torch', device=device).squeeze(0).float()

    # mean_scf_diff, max_scf_diff = rho_scf_diff.abs().mean().item(), rho_scf_diff.abs().max().item()
    # err_I_scf = eval_I(rho_scf, rho_target, mol.grid.weights)
    # print("i_struc %d,\tmean_SCF_diff = %10f\tmax_SCF_diff = %10f" %(i_struc, mean_scf_diff, max_scf_diff))
    # print("errorItg_SCF = %10.8e"%(err_I_scf))

    test_err[0,i_struc] = mean_rho_diff
    test_err[1,i_struc] = max_rho_diff
    test_err[2,i_struc] = err_I
    # test_err[3,i_struc] = mean_scf_diff
    # test_err[4,i_struc] = max_scf_diff
    # test_err[5,i_struc] = err_I_scf

    with open(osp.join(path, 'coords.pkl'), 'ab') as f_c:
        pickle.dump(data['mol'].grid.coords, f_c)
    with open(osp.join(path, 'rho_diff.pkl'), 'ab') as f_rho:
        pickle.dump(rho_diff.cpu().numpy(), f_rho)
    # with open(osp.join(path, 'rho_scf_diff.pkl'), 'ab') as f_rho_scf:
    #     pickle.dump(rho_scf_diff.cpu().numpy(), f_rho_scf)
    with open(osp.join(path, 'vxc.pkl'), 'ab') as f_v:
        pickle.dump(vxc.cpu().numpy(), f_v)

np.save(osp.join(path, 'test_err_log.npy'), test_err)