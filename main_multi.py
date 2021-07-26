from __future__ import print_function
import math
import sys
import numpy as np
import scipy

#import matplotlibc
#matplotlib.use("WXAgg")
#import matplotlib.pyplot as plt
import torch.optim as optim

from const_list import *
from Config import get_options, get_bpks_options
from dataset import *
from func import *
from nnks import *
from xc import *
from log_to_file import Logger

from dataset import gen_mesh_2, DensityToPotentialDataset

from pyscf import dft

def main():
    options = get_options(sys.argv[1])

    logger = Logger(options["log_path"], to_stdout=options["verbose"])
    logger.log("========Task Start========")

#    train_set_loader, validate_set_loader = \
#            get_train_and_validate_set(options)

    model = get_model(options["model"])()
    if "restart" in options.keys():
        load_model(model, options["restart"])
        print('loading: ', options["restart"])
    if options["enable_cuda"]:
        model.cuda()

    if options["loss_function"] == "MSELoss_zsym":
        if "zsym_coef" in options.keys():
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])(float(options["zsym_coef"]))
        else:
            loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
    else:
        loss_func = get_list_item(LOSS_FUNC_LIST, options["loss_function"])()
    loss_func.size_average = True

    optimiser = get_list_item(OPTIM_LIST, options["optimiser"])(model.parameters(), lr=options["learning_rate"])

    # start logger
    logger.log(str(model), "main")
    logger.log("Max iteration: %d" % (options["max_epoch"]), "main")
    logger.log("Learning rate: %e" % (options["learning_rate"]), "main")
    logger.log("Loss function", "main")
    logger.log(str(loss_func), "main")
    logger.log("Optimiser", "main")
    logger.log(str(optimiser), "main")
    logger.log("Model saved to %s" % (options["model_save_path"]), "main")

    n_restart = int(options["n_restart"]) if "n_restart" in options.keys() else max(1, options["max_epoch"]//10)
    #save_period = int(options["save_period"]) if "save_period" in options.keys() else 100
    n_save_itr = int(options["n_save_itr"]) if "n_save_itr" in options.keys() else 1
    n_save_iks = int(options["n_save_iks"]) if "n_save_iks" in options.keys() else 1

    # KS prepare
    bpks_opts = get_bpks_options(sys.argv[1], 'BPKS')
    # nnks = NNKS(bpks_opts)

    print("====OPTIONS====")
    for k, v in options.items():
        print(k, v, v.__class__)
    print("====BPKS====")
    for k, v in bpks_opts.items():
        print(k, v, v.__class__)

    def solve_KS(f, s, mo_occ):
        e, c = scipy.linalg.eigh(f, s)
        mocc = c[:, mo_occ>0]
        dm = numpy.dot(mocc * mo_occ[mo_occ>0], mocc.T.conj())
        return e, c, dm

    def density_on_grids(mol, coords, dm, deriv=0):
        XCTYPE = ['lda', 'gga']
        ao_values = dft.numint.eval_ao(mol, coords, deriv=deriv)
        rho = dft.numint.eval_rho(mol, ao_values, dm, xctype=XCTYPE[deriv])
        return rho   
    def density_on_grids_blk(mol, coords, dm, deriv=0):
        XCTYPE = ['lda', 'gga']
        total = len(coords)
        n_blk = total // BLK_SIZE
        res = total - n_blk * BLK_SIZE
        if deriv == 0:
            rho = np.empty(total)
        elif deriv == 1:
            rho = np.empty([4, total])
        else:
            raise NotImplementedError

        with tqdm(total=total) as pbar:
            for i_blk in range(n_blk):
                ao_values = dft.numint.eval_ao(mol, coords[i_blk * BLK_SIZE : (i_blk + 1) * BLK_SIZE], deriv=deriv)
                rho[:, i_blk * BLK_SIZE : (i_blk + 1) * BLK_SIZE] = dft.numint.eval_rho(mol, ao_values, dm, xctype=XCTYPE[deriv])
                pbar.update(BLK_SIZE)
            if res > 0:
                ao_values = dft.numint.eval_ao(mol, coords[-res:], deriv=deriv)
                rho[:, -res:] =dft.numint.eval_rho(mol, ao_values, dm, xctype=XCTYPE[deriv])
                pbar.update(res)
        return rho

    def ccsd(mol, mcc=None, mf=None):
        if mf is None:
            mf = pyscf.scf.RHF(mol)
            mf.kernel()
        c = mf.mo_coeff
        if mcc is None:
            mcc = pyscf.cc.CCSD(mf)
        ecc, t1, t2 = mcc.kernel()
        rdm1 = mcc.make_rdm1()
        rdm1_ao = einsum('pi,ij,qj->pq', c, rdm1, c.conj())
        return rdm1_ao
    # dm_ccsd = ccsd(nnks.mol)
    # rho_ccsd = density_on_grids(nnks.mol, nnks.grids.coords, dm_ccsd)

    def gen_offset(cube_length, cube_point):
        a = cube_length
        na = cube_point
        da = a / float(na - 1)
        a0 = -a / 2.
        offset = numpy.zeros([na*na*na, 3], dtype=float)
        p = 0
        for i in range(na):
            for j in range(na):
                for k in range(na):
                    offset[p][0] = a0 + da * i
                    offset[p][1] = a0 + da * j
                    offset[p][2] = a0 + da * k
                    p += 1
        return offset

    def rotate_offset(off, rot_seq):
        for rot_ax, ang in rot_seq:
            all_ax = [0,1,2]
            ox = all_ax[rot_ax+1:]+all_ax[:rot_ax]
            off[:,ox[0]], off[:,ox[1]] = off[:,ox[0]]*np.cos(ang)-off[:,ox[1]]*np.sin(ang), off[:,ox[1]]*np.cos(ang)+off[:,ox[0]]*np.sin(ang)
            return off
    
    def flip_offset(off, flip_or_not_xyz):
        for i, f, in enumerate(flip_or_not_xyz):
            off[:, i] = (1-2*f) * off[:, i]
        return off

    def gen_grids(mesh, cube_length, cube_point):
        offset = gen_offset(cube_length, cube_point)
        coords = numpy.zeros([len(mesh)*len(offset), 3], dtype=float)
        na3 = len(offset)
        assert(na3 == cube_point**3)
        for i, m in enumerate(mesh):
            coords[i * na3 : (i + 1) * na3] = m + offset
        return coords

    def gen_grids_repeated_sampling(mesh, cube_length, cube_point, nr):
        offset = gen_offset(cube_length, cube_point)
        sz_mesh = len(mesh)
        na3 = len(offset)
        assert(na3 == cube_point**3)
        coords = numpy.zeros([sz_mesh*na3*nr, 3], dtype=float)
        n_rot = 5
        for i, m in enumerate(mesh):
            for j in range(nr):
                # flipping
                flip_or_not_xyz = np.random.randint(2, size=3)
                offset = flip_offset(offset, flip_or_not_xyz)
                # rotating
                rot_seq = []
                for _ in range(n_rot):
                    rot_seq.append([np.random.randint(3), np.random.rand()*np.pi*2])
                offset = rotate_offset(offset, rot_seq)
                coords[i*na3*nr+j*na3 : i*na3*nr+(j+1)*na3] = m + offset
        return coords

    def gen_grids_repeated_sampling_flip_only(mesh, cube_length, cube_point):
        offset = gen_offset(cube_length, cube_point)
        sz_mesh = len(mesh)
        na3 = len(offset)
        assert(na3 == cube_point**3)
        #all_flips = np.array(np.meshgrid([0,1],[0,1],[0,1])).transpose(2,1,3,0).reshape(-1,3)
        all_flips = np.array([[0, 0, 0],[0, 0, 1]])
        nr = all_flips.shape[0]
        coords = numpy.zeros([sz_mesh*na3*nr, 3], dtype=float)
        for i, m in enumerate(mesh):
            for j in range(nr):
                offset = flip_offset(offset, all_flips[j])
                coords[i*na3*nr+j*na3 : i*na3*nr+(j+1)*na3] = m + offset
        return coords


    # print('generating grids')
    # coords = gen_grids(nnks.grids.coords, bpks_opts['CubeLength'], bpks_opts['CubePoint'])
    # # print(coords.shape, dm_ccsd.shape)
    # rawdata_x = density_on_grids(nnks.mol, coords, dm_ccsd, deriv=1)
    # # print(rawdata_x.shape)
    # na = bpks_opts['CubePoint']
    # rawdata_x = rawdata_x.reshape(4, len(coords) // na**3, na**3)
    # # (4, n, 729)
    # rawdata_x = np.concatenate((
    #     rawdata_x[0],
    #     rawdata_x[1],
    #     rawdata_x[2],
    #     rawdata_x[3],
    # ), axis=1)
    # rho_input = rawdata_x.reshape(-1, 4, na, na, na)


    # Train!!
    logger.log("Train start", "main")
    
    train_x = np.array(range(options["max_epoch"]))
    train_y = np.zeros(len(train_x), dtype=np.float32)
    validate_x = np.array(range(options["max_epoch"]))
    validate_y = np.zeros(len(validate_x), dtype=np.float32)

    print("generating initial net:")
    train_set_loader, validate_set_loader = get_train_and_validate_set(options)
    for epoch in range(options["max_epoch"]):
        loss_on_train_init = train(epoch, train_set_loader, model, loss_func, optimiser, logger, cuda=options["enable_cuda"])
        loss_on_validate_init = validate(epoch, validate_set_loader, model, loss_func, logger, cuda=options["enable_cuda"])

        if epoch % (n_restart) == 0:
            print("epoch: ", epoch)
            save_model(model, options["model_save_path"] + ".restart%d" % (epoch+1))
        train_y[epoch] = loss_on_train_init
        validate_y[epoch] = loss_on_validate_init

    train_x_bpks = np.array(range(options["max_itr_bpks"]))
    train_y_bpks = np.zeros(len(train_x_bpks), dtype=np.float32)
    validate_x_bpks = np.array(range(options["max_itr_bpks"]))
    validate_y_bpks = np.zeros(len(validate_x_bpks), dtype=np.float32)
   

    # prepare all structures
    print("preparing all structures and fixed data ...")
    na = bpks_opts['CubePoint']
    if bpks_opts['flipOnlyForRepeatedSampling']:
        nr = 2
        print("nRepeatedSampling is not in use, nr =  for flip only.")
    else:
        nr = bpks_opts['nRepeatedSampling']
    dist_range = np.linspace(options["bg_dist"], options["ed_dist"], options["n_struc"])
    all_nnks = []
    for struc_idx, dist in enumerate(dist_range):
        print("structure: %d, " %struc_idx)
        nnks = NNKS(bpks_opts, dimer_dist=dist, atom_list=['He','H'], charge=1, spin=0, grid_level=bpks_opts['MeshLevel'])
        nnks.dm = ccsd(nnks.mol)
        nnks.rho = density_on_grids(nnks.mol, nnks.grids.coords, nnks.dm)
        all_nnks.append(nnks)
        if nr <= 1:
            print("No rotation.")
            data_coords = gen_grids(nnks.grids.coords, bpks_opts['CubeLength'], na)
        elif bpks_opts['flipOnlyForRepeatedSampling']:
            data_coords = gen_grids_repeated_sampling_flip_only(nnks.grids.coords, bpks_opts['CubeLength'], na)
        else:
            data_coords = gen_grids_repeated_sampling(nnks.grids.coords, bpks_opts['CubeLength'], na, nr)
        rawdata_x = density_on_grids_blk(nnks.mol, data_coords, nnks.dm, deriv=1)
        if bpks_opts['dataAug'] or nr <= 1:    # by data augmentation
            rawdata_x = rawdata_x.reshape(4, len(data_coords)//na**3, na, na, na)
            rho_input = rawdata_x.transpose(1,0,2,3,4)
        else:   # by building-in model
            rawdata_x = rawdata_x.reshape(4, len(data_coords)//nr//na**3, nr, na, na, na)
            rho_input = rawdata_x.transpose(1,2,0,3,4,5)
        
        np.save('data/rho_input%d.npy' % struc_idx, rho_input)

        # # symmetry
        # # (x, y, z) -> (r, 0, z)
        # grids_xz = nnks.grids.coords.copy()
        # grids_xz[:, 0] = np.sqrt(grids_xz[:, 0] ** 2 + grids_xz[:, 1] ** 2)
        # grids_xz[:, 1] = 0.
        # grids_xz[:, 2] = np.abs(grids_xz[:, 2])
        # data_coords_xz = gen_grids(grids_xz, bpks_opts['CubeLength'], bpks_opts['CubePoint'])
        # rawdata_xz_x = density_on_grids_blk(nnks.mol, data_coords_xz, nnks.dm, deriv=1)
        # na = bpks_opts['CubePoint']
        # rawdata_xz_x = rawdata_xz_x.reshape(4, len(data_coords_xz) // na**3, na, na, na)
        # rho_input_xz = rawdata_xz_x.transpose(1,0,2,3,4)
        # np.save('data/rho_input%d.npy' % struc_idx, rho_input_xz)

    # determine counts
    epoch = 0
    # itr_save = 0
    rho_diff_file = open("rho_diff.txt", 'w')
    max_itr, n_struc, max_iks, max_sub_epoch = options["max_itr_bpks"], options["n_struc"], options["max_itr_each_struc"], options["max_sub_epoch_bpks"]
    # all_rho_diff = np.zeros([-(-max_itr*n_struc*max_iks//save_period), len(all_nnks[0].rho)])
    if max_itr == 1:
        save_period_itr = 1
        n_save_itr = 1
    elif n_save_itr == 1:
        save_period_itr = max_itr
    else:
        save_period_itr = (max_itr-1)//(n_save_itr-1)
    if max_iks == 1:
        save_period_iks = 1
        n_save_iks = 1
    elif n_save_iks == 1:
        save_period_iks = max_iks
    else:
        save_period_iks = (max_iks-1)//(n_save_iks-1)
    #save_period_itr, save_period_iks = (max_itr-1)//(n_save_itr-1), (max_iks-1)//(n_save_iks-1)

    # testing initial model
    init_rho_diff = np.zeros([n_struc, len(all_nnks[0].rho)])
    init_vxc = np.zeros([n_struc, len(all_nnks[0].rho)])
    for struc_idx in range(n_struc):
        nnks = all_nnks[struc_idx]
        rho_input = np.load('data/rho_input%d.npy' % struc_idx)
        rho_input = torch.from_numpy(rho_input).float().cuda()
        if bpks_opts['extraError'] == True:
            vxc, _ = model(rho_input)
        else:
            vxc = model(rho_input)
        print("vxc shape: ", vxc.shape)
        vxc = vxc.view(-1,1)
        if vxc.shape[0] > nnks.grids.coords.shape[0]:
            vxc = vxc.view(-1, nr)
            vxc = torch.mean(vxc, 1)
        assert vxc.shape[0] == nnks.grids.coords.shape[0], "wrong vxc shape"
        vxc = vxc.detach().cpu().numpy()
        vxc = vxc.reshape(-1)
        vxc_mat = eval_xc_mat(nnks, model=model, wxc=vxc, vxc=vxc)
        nnks.e, nnks.c, dm_new = solve_KS(nnks.H+vxc_mat, nnks.S, nnks.mo_occ)
        rho_new = density_on_grids(nnks.mol, nnks.grids.coords, dm_new)
        rho_diff = rho_new - nnks.rho
        mean_rho_diff, max_rho_diff = np.mean(np.abs(rho_diff)), np.max(np.abs(rho_diff))
        print("Initial: struc_idx = %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(struc_idx, mean_rho_diff, max_rho_diff))
        init_rho_diff[struc_idx] = rho_diff
        init_vxc[struc_idx] = vxc
    np.savez("c_init_rho_v.npz", coords=[nnks.grids.coords for nnks in all_nnks], rho_diff=init_rho_diff, vxc=init_vxc)
    
    # training
    all_rho_diff = np.zeros([n_save_itr, n_struc, n_save_iks, len(all_nnks[0].rho)])
    all_vxc = np.zeros([n_save_itr, n_struc, n_save_iks, len(all_nnks[0].rho)])
    itr_save, iks_save = 0, 0
    for itr in range(max_itr):
        for struc_idx in range(n_struc):
            nnks = all_nnks[struc_idx]
            rho_input = np.load('data/rho_input%d.npy' % struc_idx)
            rho_input = torch.from_numpy(rho_input).float().cuda()
            for iks in range(max_iks):
                #print("Forward pass ...")
                if bpks_opts['extraError'] == True:
                    vxc, _ = model(rho_input)
                else:
                    vxc = model(rho_input)
                #print("Outputs calculated.")
                vxc = vxc.view(-1,1)
                if vxc.shape[0] > nnks.grids.coords.shape[0]:
                    vxc = vxc.view(-1, nr)
                    vxc = torch.mean(vxc, 1)
                assert vxc.shape[0] == nnks.grids.coords.shape[0], "wrong vxc shape"
                vxc = vxc.detach().cpu().numpy()
                vxc = vxc.reshape(-1)
                #print("Evaluating matrices ...")
                vxc_mat = eval_xc_mat(nnks, model=model, wxc=vxc, vxc=vxc)
                #print("Solving KS ...")
                nnks.e, nnks.c, dm_new = solve_KS(nnks.H+vxc_mat, nnks.S, nnks.mo_occ)
                #print("Evaluating density on grids ...")
                rho_new = density_on_grids(nnks.mol, nnks.grids.coords, dm_new)
                rho_diff = rho_new - nnks.rho
                mean_rho_diff, max_rho_diff = np.mean(np.abs(rho_diff)), np.max(np.abs(rho_diff))
                print("itr = %d\tstruc_idx = %d\titr_ks = %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(itr, struc_idx, iks, mean_rho_diff, max_rho_diff))
                rho_diff_file.write("itr = %d\tstruc_idx = %d\titr_ks = %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f\n")
                if max_rho_diff < bpks_opts['ConvergenceCriterion']: break

                # updating
                guide_vxc = (rho_new - nnks.rho)*bpks_opts['GuideRate'] + vxc
                guide_vxc = guide_vxc - np.mean(guide_vxc)
                v_target = guide_vxc.reshape(-1, 1)
                v_target = torch.from_numpy(v_target).float().cuda()
                if bpks_opts['dataAug']:
                    v_target = v_target.repeat(1, nr)
                    v_target = v_target.view(-1, 1)
                #print("Backward pass ...")
                for sub_epoch in range(max_sub_epoch):  
                    loss_on_train = train_for_simple_bpks(epoch, int(options['batch_size']), rho_input, v_target, model, loss_func, optimiser, bpks_opts['extraError'], logger)
                    epoch += 1
                #print("Weights updated.")
                tot_iks = itr*n_struc*max_iks + struc_idx*max_iks + iks
                #if tot_iks % (save_period) == 0:    
                if itr % save_period_itr == 0 and iks % save_period_iks == 0:    
                    print("saving_model...", "itr_save = ", itr_save, "iks_save = ", iks_save)
                    all_rho_diff[itr_save][struc_idx][iks_save] = rho_diff
                    all_vxc[itr_save][struc_idx][iks_save] = vxc
                    if iks_save == n_save_iks-1:
                        iks_save = 0
                        if struc_idx == n_struc-1: 
                            itr_save += 1
                    else:
                        iks_save += 1
                    save_model(model, options["model_save_path"] + ".bpks_restart_itr%dstruc%diks%d" %(itr, struc_idx, iks))
                    #print("Model saved.")
    rho_diff_file.close()        
    np.savez("c_rho_v.npz", coords=[nnks.grids.coords for nnks in all_nnks], rho_diff=all_rho_diff, vxc=all_vxc)
    
    save_model(model, options["model_save_path"])
    logger.log("Model saved.", "main")

    logger.log("========Task Finish========")

    #plt.xlabel("epoch")
    #plt.ylabel("MSE Loss")
    #plt.plot(train_x, train_y, "bo-", label="train")
    #plt.plot(validate_x, validate_y, "ro-", label="validate")
    #plt.grid()
    #plt.legend(frameon=False)
    #plt.show()

if __name__ == "__main__":
    main()

