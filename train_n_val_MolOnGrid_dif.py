# differentiable training

from sympy.core.evalf import N
from options import gen_val_opts, read_options_yaml, print_options_yaml, define_train, compare_opts
from IO import save_check, load_check
from molecule import Molecule
from grids import SamplingCoords
from data import MolOnGrid, aprox_Hartree, collate_fn_MolOnGrid, collate_fn_MolOnGrid_w_grid_edges, eval_I, loss_rho_diff_Hartree, loss_w_grid_weights, reg_1deriv, mse
import torch
import numpy as np
import pickle
import os
import os.path as osp
from torch.utils.data import DataLoader
import gen_molecules
from opt_einsum import contract

import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import shared_memory
from torch.nn.parallel import DistributedDataParallel as DDP


def training_n_validation_process(rank, world_size, 
                                  tr_dataset, val_dataset,
                                  tr_opts, val_opts,
                                  shm_tr_dms_name, tr_dm_idces,
                                  shm_tr_err_name, shm_val_err_name, 
                                  tr_batch_size=3, val_batch_size=3):
    # setting up
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    device='cuda:'+str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    tr_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset) # sampler shuffles data
    # tr_collate_fn = collate_fn_MolOnGrid_w_grid_edges if 'curve_1deriv' in tr_opts['regulators'] else collate_fn_MolOnGrid
    tr_collate_fn = collate_fn_MolOnGrid
    tr_loader = DataLoader(tr_dataset, 
                            batch_size=tr_batch_size, shuffle=False, 
                            sampler=tr_sampler, collate_fn=tr_collate_fn,
                            pin_memory=True)  
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            batch_size=val_batch_size, shuffle=False, 
                            sampler=val_sampler, collate_fn=collate_fn_MolOnGrid,
                            pin_memory=True)  
    # define model and optimzer
    model, optimizer, loss_func = define_train(tr_opts, is_ddp=True, device_or_rank=rank)
    ini_itr, max_itr = 0, tr_opts['max_itr']
    if 'restart_from' in tr_opts and tr_opts['restart_from'] is not None:
        restart_path = tr_opts['restart_from']
        print("restarting from %s ..."%(restart_path))
        # if 'redefine_from_restart' in tr_opts:
            # if 'model' in tr_opts['redefine_from_restart']: model = None
            # if 'optimizer' in tr_opts['redefine_from_restart']: optimizer = None
        assert 'redefine_from_restart' not in tr_opts
        ini_itr, model, optimizer, ini_loss = load_check(restart_path, model=model, optimizer=optimizer)

    res_dir = tr_opts['results']['path']
    val_res_dir = val_opts['results']['path']

    # training & validation
    guide_rate = tr_opts['guide_rate']
    max_itr, max_iks = tr_opts['max_itr'], tr_opts['max_iks']
    itr_save, iks_save = tr_opts['itr_save_every'], tr_opts['iks_save_every']
    itr_save_check = tr_opts['itr_save_check_every'] if 'itr_save_check_every' in tr_opts else tr_opts['itr_save_every']
    itr_val = val_opts['itr_validate_every'] if 'itr_validate_every' in val_opts else 1
    itr_save_val = val_opts['itr_save_validate_every'] if 'itr_save_validate_every' in val_opts else tr_opts['itr_save_every']
    
    shm_tr_err = shared_memory.SharedMemory(name=shm_tr_err_name)
    tr_err = np.ndarray((2, max_itr, max_iks, len(tr_dataset)), dtype=float, buffer=shm_tr_err.buf)
    shm_val_err = shared_memory.SharedMemory(name=shm_val_err_name)
    val_err = np.ndarray((2, max_itr, 1, len(val_dataset)), dtype=float, buffer=shm_val_err.buf)
    shm_tr_dms = shared_memory.SharedMemory(name=shm_tr_dms_name)
    tr_dms = np.ndarray(tr_dm_idces[-1], dtype=float, buffer=shm_tr_dms.buf)

    for itr in range(ini_itr, ini_itr+max_itr):
        
        # training
        tr_sampler.set_epoch(itr)  # sampler reset seed
        model.train()
        for i_batch, data in enumerate(tr_loader):            
            # if 'curve_1deriv' in tr_opts['regulators']:
            #     mol_batch, x_batch, grid_sizes, ei_batch, ev_batch, edge_sizes = data.values()
            # else:
            #     mol_batch, x_batch, grid_sizes = data.values()
            data_indces, mol_batch = data['i'], data['mol']
            x_batch, grid_sizes = data['x'], data['grid_size']
            # if 'curve_1deriv' in tr_opts['regulators']:
            #     ei_batch, ev_batch, edge_sizes = data['ei'], data['ev'], data['edge_size']
            # through model
            x_batch = x_batch.to(device, non_blocking=True)
            for iks in range(max_iks):    
                optimizer.zero_grad()
                y_batch = model(x_batch)
                grid_bg = 0
                # if 'curve_1deriv' in tr_opts['regulators']: edge_bg = 0
                batch_loss = 0.0
                # structures in the batch
                for i_struc_in_batch, mol in enumerate(mol_batch):
                    i_struc = data_indces[i_struc_in_batch]
                    grid_range = range(grid_bg, grid_bg + grid_sizes[i_struc_in_batch])
                    y = y_batch[grid_range].mean(1)
                    # vxc = y.detach()
                    # # calculate new density
                    # vxc_mat = mol.V_mat(vxc, package='torch', device=device) # any package returns nparray since its just a mat
                    # # vxc_control = torch.zeros_like(vxc)
                    # # vxc_mat = mol.V_mat(vxc_control, package='torch', device=device)
                    # dm_new = mol.dm_from_Vxc_mat(vxc_mat)  # KS
                    
                    # # updates J and K too, but lagging 1 iteration
                    # dm = tr_dms[tr_dm_idces[i_struc]:tr_dm_idces[i_struc+1]].reshape(
                    #      [mol.basis.n_orbs, mol.basis.n_orbs])
                    # H = mol.gen_hf_H(dm)
                    # _, _, dm_new = mol.solve_KS(H + vxc_mat, mol.S, mol.mo_occ)
                    # tr_dm_mixing = 0.1 
                    # dm[:] = (1.0-tr_dm_mixing) * dm[:] + tr_dm_mixing * dm_new
                    # assert np.shares_memory(dm, tr_dms)

                    # J, K from dm_ccsd
                    # dm = mol.dm_ccsd
                    # H = mol.gen_hf_H(dm)
                    # _, _, dm_new = mol.solve_KS(H + vxc_mat, mol.S, mol.mo_occ)
                    
                    # differentable KS
                    vxc = y
                    phi = torch.from_numpy(mol.phi).float().to(device, non_blocking=True)
                    w = torch.from_numpy(mol.grid.weights).float().to(device, non_blocking=True)
                    vxc_mat = contract('i..., j..., ..., ... -> ij', phi, phi, vxc, w)
                    H = torch.from_numpy(mol.H).float().to(device, non_blocking=True)
                    S = torch.from_numpy(mol.S).float().to(device, non_blocking=True)
                    _, _, dm_new = Molecule.solve_KS_torch(H + vxc_mat, S, mol.mo_occ)
                    dm_target = torch.from_numpy(mol.dm_ccsd).float().to(device, non_blocking=True)
                    dm_diff = dm_new - dm_target
                    rho_diff = contract('i..., j..., ij->...', phi, phi, dm_diff).squeeze(0)
                    # batch_loss += loss_w_grid_weights(rho_diff, 0.0, w, device)
                    # batch_loss += mse(rho_diff, 0.0)
                    # batch_loss += mse(dm_diff, 0.0)
                    itg_2e = torch.from_numpy(mol.itg_2e).float().to(device, non_blocking=True)
                    batch_loss += loss_rho_diff_Hartree(dm_diff, itg_2e, device)


                    # rho_new = mol.density_on_grid(dm_new, deriv=0, package='torch', device=device).squeeze(0).float()                    
                    # # use new density for loss
                    # # rho_diff = rho_new - mol.rho('ccsd', package='torch', device=device).squeeze(0).float()
                    # rho_target = torch.from_numpy(mol.rho_ccsd).float().to(device, non_blocking=True)
                    # rho_diff = rho_new - rho_target
                    # benchmark
                    mean_rho_diff, max_rho_diff = rho_diff.abs().mean().item(), rho_diff.abs().max().item()                   
                    tr_err[0, itr-ini_itr, iks, i_struc], tr_err[1, itr-ini_itr, iks, i_struc] = mean_rho_diff, max_rho_diff
                    # i_struc = mol.custom_description['index']
                    print("itr %d,istruc %d,iks %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(itr, i_struc, iks, mean_rho_diff, max_rho_diff))
                    
                    # # WY
                    # guide_rate_wy = 10.0
                    # wy = guide_rate_wy * rho_diff
                    # # # ZMP
                    # # guide_rate_zmp = 1000.0
                    # # zmp_fake_Htr = aprox_Hartree(rho_diff, mol.grid, device)
                    # # zmp = guide_rate_zmp * zmp_fake_Htr
                    # # # VLB
                    # # guide_rate_vlb = 1000
                    # # # vlb = guide_rate_vlb * (rho_new/rho_target-1) * (vxc-2.0*vxc.max()+vxc.min()).abs()
                    # # vlb = guide_rate_vlb * (rho_new/rho_target-1) * (vxc-vxc.max()).abs()
                    
                    # target = vxc + wy

                    # # target = y.detach() + guide_rate * rho_diff
                    # target -= target.mean()

                    # batch_loss += loss_func(y, target)
                    # # batch_loss += loss_w_grid_weights(y, target, mol.grid.weights, device)
                    # # batch_loss += mse(y, target)
                    # # reg
                    # # if 'curve_1deriv' in tr_opts['regulators']:
                    # #     # edge_range = range(edge_bg, edge_bg + edge_sizes[i_struc_in_batch])
                    # #     # curve_reg_1deriv = tr_opts['regulators']['curve_1deriv']['strength'] * \
                    # #     #                    ( (y[ei_batch[1][edge_range]] - y[ei_batch[0][edge_range]])**2 / \
                    # #     #                      (ev_batch[edge_range]**2).sum(-1) ).sum()
                    # #     # curve_reg_1deriv = tr_opts['regulators']['curve_1deriv']['strength'] * \
                    # #     #                    ( (y[mol.grid.edge_idx[0]] - y[mol.grid.edge_idx[1]])**2 / \
                    # #     #                      (mol.grid.edge_vec**2).sum(-1) ).sum()
                    # #     curve_reg_1deriv = reg_1deriv(y, mol.grid, device)
                    # #     batch_loss += curve_reg_1deriv
                    # #     # edge_bg += edge_sizes[i_struc_in_batch]
                    # # # next structure
                    grid_bg += grid_sizes[i_struc_in_batch]
                    
                    # save
                    if itr % itr_save == 0 and iks % iks_save == 0:
                        print("itr: %d\ti_struc: %d\tiks: %d\tloss: %10.3e"%(itr, i_struc, iks, batch_loss.item()))
                        with open(osp.join(res_dir, 'rho_diff%d.pkl'%(i_struc)), 'ab') as f_rho:
                            print('Saving rho at itr=%d,iks=%d...'%(itr,iks), end='')
                            pickle.dump(rho_diff.detach().cpu().numpy(), f_rho)
                        with open(osp.join(res_dir, 'vxc%d.pkl'%(i_struc)), 'ab') as f_v:
                            print('Saving V at itr=%d,iks=%d...'%(itr,iks), end='')
                            pickle.dump(vxc.detach().cpu().numpy(), f_v)
                        # save_check('./checkpoints', itr, model, optimizer, batch_loss)
                        # print('Saving ckeck at itr=%d,iks=%d...'%(itr,iks), end='')
                        print('Done.')
                # backpropagation
                batch_loss.backward()
                print('grad norm: ', torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
        # dist.barrier()

        # validation
        if itr % itr_val == 0:
            model.eval()
            # rho_diff_all_val = {'itr': itr}
            # vxc_all_val = {'itr': itr}
            for i_batch, data in enumerate(val_loader):
                # mol_batch, x_batch, grid_sizes = data.values()
                data_indces, mol_batch = data['i'], data['mol']
                x_batch, grid_sizes = data['x'], data['grid_size']
                # through model
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = model(x_batch)
                grid_bg = 0
                # structures in the batch
                for i_struc_in_batch, mol in enumerate(mol_batch):
                    i_struc = data_indces[i_struc_in_batch]
                    grid_range = range(grid_bg, grid_bg + grid_sizes[i_struc_in_batch])
                    y = y_batch[grid_range].mean(1)
                    vxc = y.detach()
                    # vxc = y.detach().to('cpu').numpy()
                    # calculate new density
                    # vxc_mat = mol.V_mat(vxc, package='numpy')
                    vxc_mat = mol.V_mat(vxc, package='torch', device=device)
                    # vxc_control = torch.zeros_like(vxc)
                    # vxc_mat = mol.V_mat(vxc_control, package='torch', device=device)
                    dm_new = mol.dm_from_Vxc_mat(vxc_mat)  # KS
                    rho_new = mol.density_on_grid(dm_new, deriv=0, package='torch', device=device).squeeze(0).float()
                    # use new density for loss
                    # rho_diff = rho_new - mol.rho('ccsd', package='torch', device=device).squeeze(0).float()
                    rho_diff = rho_new - torch.from_numpy(mol.rho_ccsd).float().to(device, non_blocking=True)
                    mean_rho_diff, max_rho_diff = rho_diff.abs().mean().item(), rho_diff.abs().max().item()
                    val_err[0, itr-ini_itr, 0, i_struc], val_err[1, itr-ini_itr, 0, i_struc] = mean_rho_diff, max_rho_diff
                    # i_struc = mol.custom_description['index']
                    print("itr %d,istruc %d,\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(itr, i_struc, mean_rho_diff, max_rho_diff))
                    # next structure
                    grid_bg += grid_sizes[i_struc_in_batch]
                    # # save to dict
                    # rho_diff_all_val[i_struc] = rho_diff.cpu().numpy() # repeated i_struc will overwight
                    # vxc_all_val[i_struc] = vxc
                    if itr % itr_save_val == 0:
                        # save to file
                        with open(osp.join(val_res_dir, 'rho_diff%d.pkl'%(i_struc)), 'ab') as f_rho:
                            pickle.dump(rho_diff.cpu().numpy(), f_rho)
                        with open(osp.join(val_res_dir, 'vxc%d.pkl'%(i_struc)), 'ab') as f_v:
                            pickle.dump(vxc.cpu().numpy(), f_v)
            dist.barrier()

        # save check
        if rank == 0 and (itr % itr_save_check == 0 or itr == ini_itr+max_itr-1):
            print('Saving ckeck at itr=%d...'%(itr), end='')
            save_check('./checkpoints', itr, model, optimizer, batch_loss)
            # with open(osp.join(val_res_dir, 'rho_diff_val.pkl'), 'ab') as f_rho:
            #     pickle.dump(rho_diff_all_val, f_rho)
            # with open(osp.join(val_res_dir, 'vxc_val.pkl'), 'ab') as f_v:
            #     pickle.dump(vxc_all_val, f_v)
            print('Done.')

    # done
    shm_tr_err.close()
    shm_val_err.close()
    shm_tr_dms.close()
    dist.barrier()
    dist.destroy_process_group()


def main():

    opts = read_options_yaml('./options.yaml')  
    print("==============OPTIONS==============")
    print_options_yaml(opts)

    # prepare all structures
    # training
    tr_opts = opts['train']
    tr_batch_size = tr_opts['batch_size']
    tr_dataset = MolOnGrid(opts['molecules'], opts['sampling'], path='./')
    tr_dataset.gen_all()
    res_dir = tr_opts['results']['path']
    for i_struc, data in enumerate(tr_dataset):
        with open(osp.join(res_dir, 'coords%d.pkl'%(i_struc)), 'wb') as f_c:
            pickle.dump(data['mol'].grid.coords, f_c)
    # validation
    val_opts = gen_val_opts(opts)
    val_batch_size = val_opts['batch_size']
    val_dataset = MolOnGrid(val_opts['molecules'], val_opts['sampling'], path='./')
    val_dataset.gen_all()
    val_res_dir = val_opts['results']['path']
    for i_struc, data in enumerate(val_dataset):
        with open(osp.join(val_res_dir, 'coords%d.pkl'%(i_struc)), 'wb') as f_c:
            pickle.dump(data['mol'].grid.coords, f_c)
        

    torch.cuda.empty_cache()

    # shared memory for density matrices
    tr_dataset_dm_sizes = np.array([data['mol'].basis.n_orbs**2 for data in tr_dataset])
    tr_dm_idces = np.insert(np.cumsum(tr_dataset_dm_sizes), 0, 0.0)
    tr_dms_tmp = np.zeros(tr_dm_idces[-1], dtype=float)
    shm_tr_dms = shared_memory.SharedMemory(create=True, size=tr_dms_tmp.nbytes)
    tr_dms = np.ndarray(tr_dms_tmp.shape, dtype=tr_dms_tmp.dtype, buffer=shm_tr_dms.buf)
    for i_struc, data in enumerate(tr_dataset):
        dm = tr_dms[tr_dm_idces[i_struc]:tr_dm_idces[i_struc+1]].reshape(
             [data['mol'].basis.n_orbs, data['mol'].basis.n_orbs])
        dm[:] = data['mol'].dm_hf
        assert np.shares_memory(dm, tr_dms)

    max_itr, max_iks = tr_opts['max_itr'], tr_opts['max_iks']
    itr_val = val_opts['itr_validate_every'] if 'itr_validate_every' in val_opts else 1
    tr_err = np.zeros((2, max_itr, max_iks, len(tr_dataset) )) 
    val_err = np.zeros((2, -(-max_itr // itr_val), 1, len(val_dataset) ))
    shm_tr_err = shared_memory.SharedMemory(create=True, size=tr_err.nbytes)
    shm_val_err = shared_memory.SharedMemory(create=True, size=val_err.nbytes)
    world_size = torch.cuda.device_count()
    print('Using', world_size, 'GPUs ...')
    mp.spawn(training_n_validation_process,
             args=(world_size, tr_dataset, val_dataset,
                   tr_opts, val_opts,
                   shm_tr_dms.name, tr_dm_idces,
                   shm_tr_err.name, shm_val_err.name, 
                   tr_batch_size, val_batch_size),
             nprocs=world_size,
             join=True)
    tr_err = np.ndarray((2, max_itr, max_iks, len(tr_dataset)), dtype=float, buffer=shm_tr_err.buf)
    val_err = np.ndarray((2, max_itr, 1, len(val_dataset)), dtype=float, buffer=shm_val_err.buf)
    np.save('tr_err_log.npy', tr_err)
    np.save('val_err_log.npy', val_err)
    shm_tr_err.unlink()
    shm_val_err.unlink()
    shm_tr_dms.unlink()

if __name__=="__main__":
    main()






            