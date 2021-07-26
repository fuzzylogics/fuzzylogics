
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

# test all structures
path = './test1'
dataset = MolOnGrid(test_opts['molecules'], test_opts['sampling'], path=path, device=device)
test_err_hf = np.zeros((6, len(dataset)))
test_err_dft = np.zeros((6, len(dataset)))
test_err_oep = np.zeros((6, len(dataset)))
for i_struc, data in enumerate(dataset):
    mol = data['mol']
    
    dft_rho = torch.from_numpy(mol.rho('dft')).float().to(device).squeeze(0)
    hf_rho = torch.from_numpy(mol.rho('hf')).float().to(device).squeeze(0)
    oep_rho = torch.from_numpy(mol.rho('oep')).float().to(device).squeeze(0)
    ccsd_rho = torch.from_numpy(mol.rho('ccsd')).float().to(device).squeeze(0)
    oep_ccsd_diff = oep_rho - ccsd_rho
    hf_ccsd_diff = hf_rho - ccsd_rho
    dft_ccsd_diff = dft_rho - ccsd_rho

    mean_hf_ccsd_diff, max_hf_ccsd_diff = hf_ccsd_diff.abs().mean().item(), hf_ccsd_diff.abs().max().item()
    print("i_struc %d,\tmean_hf_ccsd_diff = %10f\tmax_hf_ccsd_diff = %10f" %(i_struc, mean_hf_ccsd_diff, max_hf_ccsd_diff))
    err_I_hf = eval_I(hf_rho, ccsd_rho, mol.grid.weights)
    print("errorItg_hf = %10.8e"%(err_I_hf))

    mean_dft_ccsd_diff, max_dft_ccsd_diff = dft_ccsd_diff.abs().mean().item(), dft_ccsd_diff.abs().max().item()
    print("i_struc %d,\tmean_dft_ccsd_diff = %10f\tmax_dft_ccsd_diff = %10f" %(i_struc, mean_dft_ccsd_diff, max_dft_ccsd_diff))
    err_I_dft = eval_I(dft_rho, ccsd_rho, mol.grid.weights)
    print("errorItg_dft = %10.8e"%(err_I_dft))
    
    mean_oep_ccsd_diff, max_oep_ccsd_diff = oep_ccsd_diff.abs().mean().item(), oep_ccsd_diff.abs().max().item()
    print("i_struc %d,\tmean_oep_ccsd_diff = %10f\tmax_oep_ccsd_diff = %10f" %(i_struc, mean_oep_ccsd_diff, max_oep_ccsd_diff))
    err_I_oep = eval_I(oep_rho, ccsd_rho, mol.grid.weights)
    print("errorItg_oep = %10.8e"%(err_I_oep))

    # # scf
    # sampling_coords = SamplingCoords(mol.grid, test_opts['sampling'])
    # sampling_coords.mesh_out()
    # dm_scf = mol.dm_scf_ml(model, sampling_coords, mol.dm_dft, device=device, input_shape='vanilla')
    # dm_scf_diff = dm_scf-mol.dm_ccsd
    # rho_scf_diff = mol.density_on_grid(dm_scf_diff, deriv=0, package='torch', device=device).squeeze(0).float()

    # mean_scf_diff, max_scf_diff = rho_scf_diff.abs().mean().item(), rho_scf_diff.abs().max().item()
    # err_I_scf = eval_I(rho_scf_diff, rho_target, mol.grid.weights)
    # print("i_struc %d,\tmean_SCF_diff = %10f\tmax_SCF_diff = %10f" %(i_struc, mean_scf_diff, max_scf_diff))
    # print("errorItg_SCF = %10.8e"%(err_I_scf))

    test_err_hf[0,i_struc] = mean_hf_ccsd_diff
    test_err_hf[1,i_struc] = max_hf_ccsd_diff
    test_err_hf[2,i_struc] = err_I_hf

    test_err_dft[0,i_struc] = mean_dft_ccsd_diff
    test_err_dft[1,i_struc] = max_dft_ccsd_diff
    test_err_dft[2,i_struc] = err_I_dft
    
    test_err_oep[0,i_struc] = mean_oep_ccsd_diff
    test_err_oep[1,i_struc] = max_oep_ccsd_diff
    test_err_oep[2,i_struc] = err_I_oep

    # with open(osp.join(path, 'coords.pkl'), 'ab') as f_c:
    #     pickle.dump(data['mol'].grid.coords, f_c)
    # with open(osp.join(path, 'rho_diff.pkl'), 'ab') as f_rho:
    #     pickle.dump(rho_diff.cpu().numpy(), f_rho)
    # # with open(osp.join(path, 'rho_scf_diff.pkl'), 'ab') as f_rho_scf:
    # #     pickle.dump(rho_scf_diff.cpu().numpy(), f_rho_scf)
    # with open(osp.join(path, 'vxc.pkl'), 'ab') as f_v:
    #     pickle.dump(vxc.cpu().numpy(), f_v)

np.save(osp.join(path, 'test_err_log_hf.npy'), test_err_hf)
np.save(osp.join(path, 'test_err_log_dft.npy'), test_err_dft)
np.save(osp.join(path, 'test_err_log_oep.npy'), test_err_oep)