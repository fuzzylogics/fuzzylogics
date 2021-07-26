
from options import read_options_yaml, print_options_yaml, define_train
from IO import save_check, load_check
from molecule import Molecule
from grids import SamplingCoords
from data import MolOnGrid
import torch
from torch.utils.data import DataLoader



opts = read_options_yaml('options.yaml')
device = 'cuda'

print("==============OPTIONS==============")
print_options_yaml(opts)

model, optimizer = define_train(opts['train'])

# define model and optimzer
tr_opts = opts['train']
model, optimizer, loss_func = define_train(tr_opts)
ini_epoch, max_epochs = 0, tr_opts['max_epoch']
if 'restart_from' in tr_opts and tr_opts['restart_from'] is not None:
    restart_path = tr_opts['restart_from']
    if 'redefine_from_restart' in tr_opts:
        if 'model' in tr_opts['redefine_from_restart']: model = None
        if 'optimizer' in tr_opts['redefine_from_restart']: optimizer = None
    ini_itr, model, optimizer, ini_loss = load_check(restart_path, model=model, optimizer=optimizer)



# prepare all structures
# print("Preparing all structures and data ...")
mol_data = MolOnGrid(opts, './', device=device)
single_mol_loader = DataLoader(mol_data)
model = model.to(device)
# testing initial model
# for i, mol in enumerate(all_mols):
max_itr, max_iks = tr_opts['max_itr'], tr_opts['max_iks']
itr_save, iks_save = tr_opts['itr_save_every'], tr_opts['iks_save_every']
for itr in range(int_itr, int_itr+max_itr):
    for data in single_mol_loader:
        x, idx = data['x'], data['idx']
        for iks in range(max_iks):     
            # through model            
            x = x.to(device)
            x = x.squeeze(0)
            x = x.transpose(1,0)
            y = model(x)
            vxc = y.detach().mean([0, 2, 3]) # vxc shape: [mol_b, grid_b, n_rot, n_mir]
            # calculate new density
            mol = single_mol_loader.load_mol(idx)
            vxc_mat = mol.V_mat(vxc, package='torch', device=device)
            dm_new = mol.dm_from_Vxc_mat(vxc_mat)
            rho_new = mol.density_on_grid(dm_new)
            # evaluate new density
            rho_diff = rho_new - mol.rho('ccsd')
            # mean_rho_diff, max_rho_diff = np.mean(np.abs(rho_diff)), np.max(np.abs(rho_diff))
            # print("Initial: struc_idx = %d\tmean_rho_diff = %10f\tmax_rho_diff = %10f" %(struc_idx, mean_rho_diff, max_rho_diff))

            target = y + rho_diff
            loss = loss_func(y, target)
            loss.backward()
            optimizer.step()

            if itr % itr_save == 0 and iks % iks_save == 0:
                save_check(ini_itr+max_itr, model, optimizer, loss)
                print("epoch: %d\tloss: %10.3e\ttime: (train: %3.3f, eval: %3.3f)"%(epoch, total_loss, t1-t2o, t2-t1))
            