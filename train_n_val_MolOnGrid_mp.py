from re import X
from time import time
from options import gen_val_opts, read_options_yaml, print_options_yaml, define_train, define_optimizer, compare_opts
from IO import save_check, load_check
from molecule import Molecule
from grids import SamplingCoords
from data import MolOnGrid, MolOnGrid_graph, aprox_Hartree, collate_fn_MolOnGrid, collate_fn_MolOnGrid_w_grid_edges, eval_I, loss_w_grid_weights, reg_0deriv, reg_1deriv, reg_1deriv_pow, mse, reg_2deriv
import torch
import numpy as np
import pickle
import os
import os.path as osp
from torch.utils.data import DataLoader
import gen_molecules

import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import shared_memory
from torch.nn.parallel import DistributedDataParallel as DDP


def training_n_validation_process(rank, world_size, 
                                  tr_dataset, val_dataset,
                                  tr_opts, val_opts,
                                  shm_tr_loss_guide_multiplier_name,
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
    if 'use_edge' in tr_opts['model'] and tr_opts['model']['use_edge']:
        model_use_edge = True
    else: model_use_edge = False
    tr_collate_fn = collate_fn_MolOnGrid_w_grid_edges if model_use_edge else collate_fn_MolOnGrid
    val_collate_fn = collate_fn_MolOnGrid_w_grid_edges if model_use_edge else collate_fn_MolOnGrid
    tr_loader = DataLoader(tr_dataset, 
                            batch_size=tr_batch_size, shuffle=False, 
                            sampler=tr_sampler, collate_fn=tr_collate_fn,
                            pin_memory=True)  
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            batch_size=val_batch_size, shuffle=False, 
                            sampler=val_sampler, collate_fn=val_collate_fn,
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
        if 'restart_skip' in tr_opts and 'optimizer' in tr_opts['restart_skip']:
            ini_itr, model, _, ini_loss = load_check(restart_path, model=model)
            optimizer = define_optimizer(tr_opts, model)
            print('using new optimizer: ', optimizer)
        else:
            ini_itr, model, optimizer, ini_loss = load_check(restart_path, model=model, optimizer=optimizer)

    res_dir = tr_opts['results']['path']
    val_res_dir = val_opts['results']['path']

    # training & validation
    max_itr, max_iks = tr_opts['max_itr'], tr_opts['max_iks']
    itr_save, iks_save = tr_opts['itr_save_every'], tr_opts['iks_save_every']
    itr_save_check = tr_opts['itr_save_check_every'] if 'itr_save_check_every' in tr_opts else tr_opts['itr_save_every']
    itr_val = val_opts['itr_validate_every'] if 'itr_validate_every' in val_opts else 1
    itr_save_val = val_opts['itr_save_validate_every'] if 'itr_save_validate_every' in val_opts else tr_opts['itr_save_every']
    
    # shared memory for errors
    shm_tr_err = shared_memory.SharedMemory(name=shm_tr_err_name)
    tr_err = np.ndarray((3, max_itr, max_iks, len(tr_dataset)), dtype=float, buffer=shm_tr_err.buf)
    shm_val_err = shared_memory.SharedMemory(name=shm_val_err_name)
    val_err = np.ndarray((3, max_itr, 1, len(val_dataset)), dtype=float, buffer=shm_val_err.buf)
    
    # shared memory for density matrices
    shm_tr_dms = shared_memory.SharedMemory(name=shm_tr_dms_name)
    tr_dms = np.ndarray(tr_dm_idces[-1], dtype=float, buffer=shm_tr_dms.buf)

    # shared memory for guide multipliers
    shm_guide_multiplier = shared_memory.SharedMemory(name=shm_tr_loss_guide_multiplier_name)
    guide_multiplier = np.ndarray(len(tr_dataset), dtype=float, buffer=shm_guide_multiplier.buf)
    guide_multiplier[:] = 1.0

    for itr in range(ini_itr, ini_itr+max_itr):
        t0 = time()
        # training
        tr_sampler.set_epoch(itr)  # sampler reset seed
        model.train()
        for i_batch, data in enumerate(tr_loader):            

            data_indces, mol_batch = data['i'], data['mol']
            x_batch, grid_sizes = data['x'], data['grid_size']            
            x_batch = x_batch.to(device, non_blocking=True)
            if model_use_edge:
                ei_batch, ev_batch, edge_sizes = data['ei'], data['ev'], data['edge_size']
                ei_batch = ei_batch.to(device, non_blocking=True)
                ev_batch = ev_batch.to(device, non_blocking=True)

            for iks in range(max_iks):    
                # pass through model
                if model_use_edge:
                    if 'output_edge' in tr_opts['data'] and tr_opts['data']['output_edge']:
                        y_batch, e_out_batch = model(x_batch, ei_batch, ev_batch)
                    else: y_batch = model(x_batch, ei_batch, ev_batch)
                else: y_batch = model(x_batch)
                # initialize batch
                grid_bg, edge_bg = 0, 0
                batch_loss = torch.tensor(0.0, device=device)
                optimizer.zero_grad()

                # structures in the batch
                for i_struc_in_batch, mol in enumerate(mol_batch):
                    
                    # indices
                    i_struc = data_indces[i_struc_in_batch]
                    grid_range = range(grid_bg, grid_bg + grid_sizes[i_struc_in_batch])                    
                    if model_use_edge: edge_range = range(edge_bg, edge_bg + edge_sizes[i_struc_in_batch])

                    # output
                    y = y_batch[grid_range].mean(1)
                   
                    # KS v to rho
                    vxc = y.detach()
                    vxc_mat = mol.V_mat(vxc, package='torch', device=device)
                    dm_new = mol.dm_from_Vxc_mat(vxc_mat)  # KS                    
                    rho_new = mol.density_on_grid(dm_new, deriv=0, package='torch', device=device).squeeze(0).float()
                    rho_target = torch.from_numpy(mol.rho_ccsd).float().to(device, non_blocking=True)
                    rho_diff = rho_new - rho_target

                    # benchmark
                    mean_rho_diff, max_rho_diff = rho_diff.abs().mean().item(), rho_diff.abs().max().item()                   
                    tr_err[0, itr-ini_itr, iks, i_struc], tr_err[1, itr-ini_itr, iks, i_struc] = mean_rho_diff, max_rho_diff                   
                    print("itr %d, istruc %d, iks %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(itr, i_struc, iks, mean_rho_diff, max_rho_diff))
                    err_I = eval_I(rho_new, rho_target, mol.grid.weights)
                    tr_err[2, itr-ini_itr, iks, i_struc] = err_I
                    print("itr %d, errorItg = %10.8e"%(itr, err_I))

                    raw_loss = 0.0
                    # loss guides
                    target = vxc.clone()
                    if 'wy' in tr_opts['loss_guides']: # WY
                        target += tr_opts['loss_guides']['wy'] * rho_diff
                    if 'vlb' in tr_opts['loss_guides']: # vlb
                        target += tr_opts['loss_guides']['vlb'] * \
                                  (rho_new/rho_target-1) * (vxc-vxc.max()).abs()
                    if 'zmp' in tr_opts['loss_guides']: # zmp
                        zmp_fake_Htr = aprox_Hartree(rho_diff, mol.grid, device)
                        target += tr_opts['loss_guides']['zmp'] * zmp_fake_Htr                   
                    if 'normalize' in tr_opts['loss_guides'] and tr_opts['loss_guides']['normalize'] == 'mean':
                        target -= target.mean()               
                    raw_loss += loss_func(y, target)

                    # regulation
                    reg_loss = 0.0
                    if 'curve_1deriv' in tr_opts['regulators']:
                        curve_reg_1deriv = tr_opts['regulators']['curve_1deriv']['strength'] * \
                                           reg_1deriv_pow(y, mol.grid, device, 
                                           tr_opts['regulators']['curve_1deriv']['power'])
                        reg_loss += curve_reg_1deriv
                    if 'curve_2deriv' in tr_opts['regulators']:
                        curve_reg_2deriv = tr_opts['regulators']['curve_2deriv']['strength'] * \
                                           reg_2deriv(y, mol.grid, device)
                        reg_loss += curve_reg_2deriv
                    if 'enforce_020' in tr_opts['regulators']:
                        if model_use_edge:
                            if 'output_edge' in tr_opts['data'] and tr_opts['data']['output_edge']:
                                y_0, e_out_0 = model(torch.zeros_like(x_batch[grid_range]), 
                                                     ei_batch[:, edge_range], ev_batch[edge_range])
                            else: y_0  = model(torch.zeros_like(x_batch[grid_range]), 
                                               ei_batch[:, edge_range], ev_batch[edge_range])
                        else: 
                            y_0 = model(torch.zeros_like(x_batch[grid_range]))
                            print('x_0 in reg020: ', torch.zeros_like(x_batch[grid_range]).max(), torch.zeros_like(x_batch[grid_range]).min())
                            print('y_0 in reg020: ', y_0.max(), y_0.min())
                        curve_reg_020 = tr_opts['regulators']['enforce_020']['strength'] * \
                                        mse(y_0)
                        print('i,', i_struc, 'reg_loss before 020 = ', reg_loss.item())
                        reg_loss += curve_reg_020
                        print('i,', i_struc, 'reg_loss after 020 = ', reg_loss.item())
                        print('i,', i_struc, 'raw_loss = ', raw_loss.item())
                    # loss_ratio = reg_loss.item() / raw_loss.item()
                    # print('i,', i_struc, 'loss_ratio, ', loss_ratio)
                   
                    # # scale losses
                    # if raw_loss.item() < 1.0:
                    #     guide_multiplier[i_struc] = 1.0 / raw_loss.item()
                    # print('i,', i_struc, 'guide_multiplier, ', guide_multiplier[i_struc])
                    # batch_loss += np.mean(guide_multiplier)*raw_loss + reg_loss
                    batch_loss += raw_loss + reg_loss

                    # save
                    if itr % itr_save == 0 and iks % iks_save == 0:
                        print("itr: %d\ti_struc: %d\tiks: %d\tloss: %10.3e"%(itr, i_struc, iks, batch_loss.item()))
                        with open(osp.join(res_dir, 'rho_diff%d.pkl'%(i_struc)), 'ab') as f_rho:
                            print('Saving rho at itr=%d,iks=%d...'%(itr,iks), end='')
                            pickle.dump(rho_diff.cpu().numpy(), f_rho)
                        with open(osp.join(res_dir, 'vxc%d.pkl'%(i_struc)), 'ab') as f_v:
                            print('Saving V at itr=%d,iks=%d...'%(itr,iks), end='')
                            pickle.dump(vxc.cpu().numpy(), f_v)
                        # save_check('./checkpoints', itr, model, optimizer, batch_loss)
                        # print('Saving ckeck at itr=%d,iks=%d...'%(itr,iks), end='')               
                        print('Done.')

                    # prepare for next struc 
                    grid_bg += grid_sizes[i_struc_in_batch]
                    if model_use_edge: edge_bg += edge_sizes[i_struc_in_batch]
                
                # backpropagation
                batch_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0e+6)
                print('grad norm: ', grad_norm.item())
                optimizer.step()
        # dist.barrier()

        # validation
        if itr % itr_val == 0:
            model.eval()
            for i_batch, data in enumerate(val_loader):
                # prepare data
                data_indces, mol_batch = data['i'], data['mol']
                x_batch, grid_sizes = data['x'], data['grid_size']            
                x_batch = x_batch.to(device, non_blocking=True)
                if model_use_edge:
                    ei_batch, ev_batch, edge_sizes = data['ei'], data['ev'], data['edge_size']
                    ei_batch = ei_batch.to(device, non_blocking=True)
                    ev_batch = ev_batch.to(device, non_blocking=True)
                # pass through model
                if model_use_edge:
                    if 'output_edge' in tr_opts['data'] and tr_opts['data']['output_edge']:
                        y_batch, e_out_batch = model(x_batch, ei_batch, ev_batch)
                    else: y_batch = model(x_batch, ei_batch, ev_batch)
                else: y_batch = model(x_batch)
                # initialize batch
                grid_bg, edge_bg = 0, 0

                # structures in the batch
                for i_struc_in_batch, mol in enumerate(mol_batch):

                    # indices
                    i_struc = data_indces[i_struc_in_batch]
                    grid_range = range(grid_bg, grid_bg + grid_sizes[i_struc_in_batch])                    
                    if model_use_edge: edge_range = range(edge_bg, edge_bg + edge_sizes[i_struc_in_batch])
                    
                    # output
                    y = y_batch[grid_range].mean(1)

                    # KS                
                    vxc = y.detach()
                    vxc_mat = mol.V_mat(vxc, package='torch', device=device)
                    dm_new = mol.dm_from_Vxc_mat(vxc_mat)  # KS
                    rho_new = mol.density_on_grid(dm_new, deriv=0, package='torch', device=device).squeeze(0).float()
                    rho_target = torch.from_numpy(mol.rho_ccsd).float().to(device, non_blocking=True)
                    rho_diff = rho_new - rho_target
                    
                    # benchmark
                    mean_rho_diff, max_rho_diff = rho_diff.abs().mean().item(), rho_diff.abs().max().item()
                    val_err[0, itr-ini_itr, 0, i_struc], val_err[1, itr-ini_itr, 0, i_struc] = mean_rho_diff, max_rho_diff
                    print("eval itr %d, istruc %d,\tmean_rho_diff = %10f\tmax_rho_diff = %10f" \
                          %(itr+1, i_struc, mean_rho_diff, max_rho_diff)) # eval itr is 1 ahead of tr itr                   
                    err_I = eval_I(rho_new, rho_target, mol.grid.weights)
                    val_err[2, itr-ini_itr, 0, i_struc] = err_I
                    print("eval itr %d, errorItg = %10.8e"%(itr+1, err_I))                   

                    # save
                    if itr % itr_save_val == 0:
                        # save to file
                        with open(osp.join(val_res_dir, 'rho_diff%d.pkl'%(i_struc)), 'ab') as f_rho:
                            pickle.dump(rho_diff.cpu().numpy(), f_rho)
                        with open(osp.join(val_res_dir, 'vxc%d.pkl'%(i_struc)), 'ab') as f_v:
                            pickle.dump(vxc.cpu().numpy(), f_v)
                    
                    # next structure
                    grid_bg += grid_sizes[i_struc_in_batch]
            
            dist.barrier()

        # save check
        if rank == 0 and (itr % itr_save_check == 0 or itr == ini_itr+max_itr-1):
            print('Saving ckeck at itr=%d...'%(itr+1), end='')
            save_check('./checkpoints', itr+1, model, optimizer, batch_loss)
            # with open(osp.join(val_res_dir, 'rho_diff_val.pkl'), 'ab') as f_rho:
            #     pickle.dump(rho_diff_all_val, f_rho)
            # with open(osp.join(val_res_dir, 'vxc_val.pkl'), 'ab') as f_v:
            #     pickle.dump(vxc_all_val, f_v)
            tr_err_filled = tr_err[:, :itr-ini_itr+1,:,:]
            val_err_filled = val_err[:, :itr-ini_itr+1,:,:]
            np.save('tr_err_log.npy', tr_err_filled)
            np.save('val_err_log.npy', val_err_filled)
            print('Done.')
        if rank == 0:
            t1 = time()
            print("Time for iteration %d is %fs."%(itr+1, t1-t0))

    # done
    shm_tr_err.close()
    shm_val_err.close()
    shm_tr_dms.close()
    shm_guide_multiplier.close()
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
    if 'save_mol_in_mem' in tr_opts and tr_opts['save_mol_in_mem']==False:
        tr_save_mol_in_mem = False
    else: tr_save_mol_in_mem = True
    if 'use_edge' in tr_opts['model'] and tr_opts['model']['use_edge']:
        tr_dataset = MolOnGrid_graph(opts['sampling']['nnn'], opts['molecules'], 
                                     path='./', 
                                     save_mol_in_mem=tr_save_mol_in_mem)
    else:
        tr_dataset = MolOnGrid(opts['molecules'], opts['sampling'], 
                               path='./', save_mol_in_mem=tr_save_mol_in_mem)
    tr_dataset.gen_all()
    res_dir = tr_opts['results']['path']
    for i_struc, data in enumerate(tr_dataset):
        with open(osp.join(res_dir, 'coords%d.pkl'%(i_struc)), 'wb') as f_c:
            pickle.dump(data['mol'].grid.coords, f_c)
    # validation
    val_opts = gen_val_opts(opts)
    val_batch_size = val_opts['batch_size']
    if 'save_mol_in_mem' in val_opts and val_opts['save_mol_in_mem']==False:
        val_save_mol_in_mem = False
    else: val_save_mol_in_mem = True
    if 'use_edge' in tr_opts['model'] and tr_opts['model']['use_edge']:
        val_dataset = MolOnGrid_graph(val_opts['sampling']['nnn'], val_opts['molecules'],
                                      path='./', 
                                      save_mol_in_mem=val_save_mol_in_mem)
    else:
        val_dataset = MolOnGrid(val_opts['molecules'], val_opts['sampling'],
                                path='./', save_mol_in_mem=val_save_mol_in_mem)
    val_dataset.gen_all()
    val_res_dir = val_opts['results']['path']
    for i_struc, data in enumerate(val_dataset):
        with open(osp.join(val_res_dir, 'coords%d.pkl'%(i_struc)), 'wb') as f_c:
            pickle.dump(data['mol'].grid.coords, f_c)
    
    torch.cuda.empty_cache()

    # shared_memory for loss guide multipliers
    tr_loss_guide_multiplier = np.ones(len(tr_dataset), dtype=float)
    shm_tr_loss_guide_multiplier = shared_memory.SharedMemory(create=True, size=tr_loss_guide_multiplier.nbytes)
    # shared memory for density matrices
    tr_dataset_dm_sizes = np.array([data['mol'].basis.n_orbs**2 for data in tr_dataset])
    tr_dm_idces = np.insert(np.cumsum(tr_dataset_dm_sizes), 0, 0.0)
    # print('Size of shared memory for DMs:', tr_dm_idces[-1])
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
    tr_err = np.zeros((3, max_itr, max_iks, len(tr_dataset) ), dtype=float) 
    val_err = np.zeros((3, -(-max_itr // itr_val), 1, len(val_dataset) ), dtype=float)
    shm_tr_err = shared_memory.SharedMemory(create=True, size=tr_err.nbytes)
    shm_val_err = shared_memory.SharedMemory(create=True, size=val_err.nbytes)
    world_size = torch.cuda.device_count()
    print('Using', world_size, 'GPUs ...')
    mp.spawn(training_n_validation_process,
             args=(world_size, tr_dataset, val_dataset,
                   tr_opts, val_opts,
                   shm_tr_loss_guide_multiplier.name,
                   shm_tr_dms.name, tr_dm_idces,
                   shm_tr_err.name, shm_val_err.name, 
                   tr_batch_size, val_batch_size),
             nprocs=world_size,
             join=True)
    tr_err = np.ndarray((3, max_itr, max_iks, len(tr_dataset)), dtype=float, buffer=shm_tr_err.buf)
    val_err = np.ndarray((3, max_itr, 1, len(val_dataset)), dtype=float, buffer=shm_val_err.buf)
    np.save('tr_err_log.npy', tr_err)
    np.save('val_err_log.npy', val_err)
    shm_tr_err.unlink()
    shm_val_err.unlink()
    shm_tr_dms.unlink()
    shm_tr_loss_guide_multiplier.unlink()


if __name__=="__main__":
    main()






            