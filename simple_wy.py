# simple wu-yang oep
#%%

import opt_einsum
import numpy as np
import scipy

def clean_array(A, crit=1e-16): 
    A[abs(A) < crit] = 0

def solve_KS(b, integrals_3c1e, H, S, mo_occ):
    V_oep = opt_einsum.contract('ijt,t->ij', integrals_3c1e, b)
    F = H + V_oep
    e, c = scipy.linalg.eigh(F, S)   
    mocc = c[:, mo_occ>0]
    dm = np.dot(mocc * mo_occ[mo_occ>0], mocc.T.conj())
    return e, c, dm, V_oep

def Grad(dm_diff, integrals_3c1e):
    g = opt_einsum.contract('ij,ijt->t', dm_diff, integrals_3c1e)
    clean_array(g)
    return g

def Hess(C, E, n_occ, integrals_3c1e):
    V_in_MO = opt_einsum.contract('ui,uvb,vx->ixb', C[:,:n_occ], integrals_3c1e, C[:,n_occ:])
    E_diff_inv = 1 / ( np.outer(E[:n_occ], np.ones_like(E)[n_occ:]) - np.outer(np.ones_like(E)[:n_occ], E[n_occ:]) )
    hess = 4.0 * opt_einsum.contract('ixb,ixd,ix->bd', V_in_MO, V_in_MO, E_diff_inv)
    return hess

def line_search(vb, grad, p, integrals_3c1e, H, S, mo_occ, dm_in):
    conv_crit = 1.e-6
    alpha = 1.e-4

    slope = -np.dot(grad, grad)
    f_old = -0.5 * slope
    lambda1 = 1.0
    vb_new = np.empty(vb.shape, dtype=float)
    f2 = 0.0; lambda2 = 0.0
    while True:
        # print("max_b=", np.max(vb))
        # print("max p=", np.max(p)) 
        # print("lambda1=", lambda1)
        vb_new = vb + lambda1 * p
        # print("sum_b_new=", np.sum(vb_new))
        _, _, dm, _ = solve_KS(
                vb_new, 
                integrals_3c1e, 
                H, S, 
                mo_occ)    
        g_new = Grad(dm-dm_in, integrals_3c1e)
        f_new = 0.5 * np.dot(g_new, g_new)
        #print("g_new", np.sum(np.abs(g_new)), "f_new", f_new, "f_old=", f_old, "slope=", slope)
        if lambda1 < conv_crit:
            return vb_new, lambda1, g_new, True
        if f_new <= f_old + alpha * lambda1 * slope:
            # print("f_old=", f_old, "slope=", slope)
            return vb_new, lambda1, g_new, False
        if lambda1 == 1.0:
            tmp_lambda1 = -slope / (2.0 * (f_new - f_old - slope))
        else:
            rhs1 = f_new - f_old - lambda1 * slope
            rhs2 = f2 - f_old - lambda2 * slope
            a = (rhs1 / (lambda1 * lambda1) - rhs2 / (lambda2 * lambda2)) / (lambda1 - lambda2)
            b = (-lambda2 * rhs1 / (lambda1 * lambda1) + lambda1 * rhs2 / (lambda2 * lambda2)) / (lambda1 - lambda2)
            if abs(a) < 1.e-10:
                tmp_lambda1 = -slope / (2.0 * b)
            else:
                disc = b * b - 3.0 * a * slope
                if disc < 0.0:
                    tmp_lambda1 = 0.5 * lambda1
                elif b <= 0.0:
                    tmp_lambda1 = (-b + np.sqrt(disc)) / (3.0 * a)
                else:
                    tmp_lambda1 = -slope / (b + np.sqrt(disc))
                if tmp_lambda1 > 0.5 * lambda1:
                    tmp_lambda1 = 0.5 * lambda1
        lambda2 = lambda1
        f2 = f_new
        lambda1 = max(tmp_lambda1, 0.1 * lambda1)

def solve_SVD(A, b, cut=1.0e-10):
    cut_value = 0.0
    #cut_value = 1.0/cut
    e, P = np.linalg.eigh(A)
    e_inv = np.where(np.abs(e)>cut, 1.0/e, cut_value)
    invA = opt_einsum.contract('ik,k,jk->ij', P, e_inv, P)
    invA = (invA + invA.T) * 0.5
    x = np.matmul(invA, b)
    return x   

def oep_search(dm_target, H, S, ne, integrals_3c1e, max_itr, oep_crit):
    mem = 0
    print('Start searching...')
    print('# iter      sum abs grad         max abs grad            # elec    lambda')
    print('--------------------------------------------------------------------------------')
    n = len(S)
    b = np.zeros(n)
    mo_occ = np.zeros(n)
    mo_occ[:ne//2] = 2
    n_occ = ne//2
    converged = False
    for i in range(max_itr):       
        clean_array(b)
        E, C, dm, _ = solve_KS(
                b,
                integrals_3c1e, 
                H, S, 
                mo_occ)
        g = Grad(dm-dm_target, integrals_3c1e)      
        print('%5d\t%16.8e\t%16.8e\t%.6lf' % 
                (
                    i+1, 
                    np.sum(abs(g)), 
                    np.max(abs(g)), 
                    np.einsum('ij,ji->', dm, S),
                    ), 
                end='')     
        if np.all(abs(g) < oep_crit):
            converged = True
            print()
            break       
        hess = Hess(C, E, n_occ, integrals_3c1e)
        #print("Hs, gd", np.sum(hess), np.sum(g))
        p = -solve_SVD(hess, g, 1.0e-10)
        clean_array(p)
        if i != 0:
            p = (1-mem) * p + mem * old_p       
        b, update_lambda, g, stop_iter = line_search(b, g, p, integrals_3c1e, H, S, mo_occ, dm_target)
        #print(b)   
        if stop_iter:
            print('\n\tSeems to find solution. Check...')
            if np.all(abs(g) < oep_crit): 
                converged = True
            else:
                print('\tFail...Halt...')
                converged = False
            break
        else:
            print('%12.4e' % (update_lambda))
        old_p = p
              
    if converged: 
        print('OEP search converged.')
    else:
        print('OEP search failed.')
    return b


#%%

if __name__ == '__main__':
    from molecule import Molecule
    # # mol_h2o = Molecule(struc_path='./H2O.str', basis_name='aug-cc-pvqz')
    # from gen_molecules import gen_water_mols_sym
    # ang = 104.15 / 180 * np.pi
    # bd = 0.9584
    # mol_h2o = gen_water_mols_sym([ang], [bd], 'aug-cc-pvqz')[0]
    # mol_h2o.grid = 3
    # from pyscf.dft import numint
    # # pyscf_phi = numint.eval_ao(mol_h2o.pyscf_mol, mol_h2o.grid_coords, deriv=0)
    # # phi_diff = mol_h2o.phi.squeeze().T - pyscf_phi
    # # max_phi_diff = np.max(phi_diff, 0)
    # # print(phi_diff.min())
    
    # import time


    # mol_h2o.dm_ccsd

    # t0=time.time()
    # ao_values = numint.eval_ao(mol_h2o.pyscf_mol, mol_h2o.grid_coords, deriv=0)
    # t1=time.time()
    # print('pyscf-phi: ', t1-t0)


    # rho_ccsd_pyscf = numint.eval_rho(mol_h2o.pyscf_mol, ao_values, mol_h2o.dm_ccsd, xctype='lda')
    # t2=time.time()
    # print('pyscf-rho: ', t2-t1)
    

    # mol_h2o.phi
    # t3 = time.time()
    # print('mol-phi: ', t3-t2)
    # rho_ccsd = mol_h2o.rho('ccsd')
    # t4 = time.time()
    # print('mol-rho: ', t4-t3)
    
    import gen_molecules
    eql_hch = 116.133 / 180 * np.pi
    eql_ch = 1.111
    eql_co = 1.205
    mol = Molecule(struc_dict = gen_molecules.gen_sym_formaldehyde_struc_dict(eql_hch, eql_ch, eql_co), 
                   basis_name='aug-cc-pvdz')
    # eql_hh = 0.7414
    # mol = Molecule(struc_dict = gen_molecules.gen_H2_struc_dict(eql_hh),
    #                basis_name = 'aug-cc-pvdz')
    mol.grid = 3
    dft_rho = mol.rho('dft')
    hf_rho = mol.rho('hf')
    hf_fixed_rho = mol.rho('hf_fixed')
    oep_rho = mol.rho('oep')
    ccsd_rho = mol.rho('ccsd')
    oep_ccsd_diff = oep_rho - ccsd_rho
    print(oep_ccsd_diff.shape)
    hf_ccsd_diff = hf_rho - ccsd_rho
    hf_fixed_ccsd_diff = hf_fixed_rho - ccsd_rho
    hf_fixed_hf_diff = hf_fixed_rho - hf_rho
    dft_ccsd_diff = dft_rho - ccsd_rho
    print('max_err_oep:', np.abs(oep_ccsd_diff).max())
    print('max_err_hf:', np.abs(hf_ccsd_diff).max())
    print('max_err_dft:', np.abs(dft_ccsd_diff).max())
    print('max_err_hf_fixed:', np.abs(hf_fixed_ccsd_diff).max())
    print('max_err_hf_fixed_to_hf:', np.abs(hf_fixed_hf_diff).max())
    # print('max_hf_dft:', np.abs(dft_rho-hf_rho).max())


    from pyscf.dft import numint
    pyscf_phi = numint.eval_ao(mol.pyscf_mol, mol.grid_coords, deriv=0)
    phi_diff = mol.phi.squeeze().T - pyscf_phi
    max_phi_diff = np.max(phi_diff, 0)
    print(phi_diff.min())


    # print(np.abs(rho_ccsd - rho_ccsd_pyscf).max())

    # mol_h2o.grid = 9
    # dft_rho = mol_h2o.rho('dft')
    # hf_rho = mol_h2o.rho('hf')
    # oep_rho = mol_h2o.rho('oep')
    # ccsd_rho = mol_h2o.rho('ccsd')
    # oep_ccsd_diff = oep_rho - ccsd_rho
    # hf_ccsd_diff = hf_rho - ccsd_rho
    # dft_ccsd_diff = dft_rho - ccsd_rho
    # print('max_err_oep:', np.abs(oep_ccsd_diff).max())
    # print('max_err_hf:', np.abs(hf_ccsd_diff).max())
    # print('max_err_dft:', np.abs(dft_ccsd_diff).max())
    # print('max_hf_dft:', np.abs(dft_rho-hf_rho).max())
    


#%%
    from find_coords import find_idx
    
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ix, x = find_idx(mol.grid_coords,0)
    # axes[0].plot(x, oep_ccsd_diff[0][ix])
    # axes[0].plot(x, hf_ccsd_diff[0][ix])
    # axes[0].plot(x, dft_ccsd_diff[0][ix])

    ixyz, xyz = find_idx(mol.grid_coords, 2)
    ax.plot(xyz, oep_ccsd_diff[0][ixyz])
    ax.plot(xyz, hf_ccsd_diff[0][ixyz])
    ax.plot(xyz, hf_fixed_ccsd_diff[0][ixyz])
    ax.plot(xyz, dft_ccsd_diff[0][ixyz])
    plt.xlim([-4, 2])
    ax.legend(['oep', 'hf', 'hf_fixed', 'b3lyp'])
    fig.savefig('ch2o_oep_hf_b3lyp.png', dpi=300)

# %%
