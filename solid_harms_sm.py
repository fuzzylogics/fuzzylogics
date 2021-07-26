# basis
from symbolic_math import flatten_poly
import sympy as sm
import numpy as np
import time

from sympy.utilities.iterables import flatten

def find_IJVszU(ang):
    # from numpy import sqrt
    from sympy import exp, sqrt, Matrix
    import re

    I_s = np.array([0])
    J_s = np.array([0])
    V_s = Matrix([1.0])
    sz_s = (1, 1)
    U_s = '1.0'

    I_p = np.array([0, 1, 2])
    J_p = np.array([0, 1, 2])
    V_p = Matrix([1.0, 1.0, 1.0])
    sz_p = (3, 3)
    U_p = 'x, y, z'

    I_d = np.array([1, 1, 1, 2, 3, 4, 4, 5])-1
    J_d = np.array([1, 4, 6, 3, 5, 1, 4, 2])-1
    V_d = Matrix([-1/2, -1/2, 1, 1, 1, sqrt(3)/2, -sqrt(3)/2, 1])
    sz_d = (5, 6)
    U_d = 'x*x, sqrt(3)*x*y, sqrt(3)*x*z, y*y, sqrt(3)*y*z, z*z'

    I_f = np.array(
        [1, 1, 1, 
         2, 2, 2, 
         3, 3, 3, 
         4, 4, 
         5, 
         6, 6, 
         7, 7]) - 1
    J_f = np.array(
        [3, 8, 10, 
         1, 4, 6, 
         2, 7, 9, 
         3, 8, 
         5, 
         1, 4, 
         2, 7]) - 1
    V_f = Matrix(
        [-3*sqrt(5)/10, -3*sqrt(5)/10, 1,
         -sqrt(6)/4, -sqrt(30)/20, sqrt(30)/5,
         -sqrt(30)/20, -sqrt(6)/4, sqrt(30)/5,
         sqrt(3)/2, -sqrt(3)/2,
         1,
         sqrt(10)/4, -3*sqrt(2)/4,
         3*sqrt(2)/4, -sqrt(10)/4])
    sz_f = (7, 10)
    U_f = re.sub('\n', ' ', '''
        x*x*x, sqrt(5)*x*x*y, sqrt(5)*x*x*z, sqrt(5)*x*y*y, sqrt(15)*x*y*z,
        sqrt(5)*x*z*z, y*y*y, sqrt(5)*y*y*z, sqrt(5)*y*z*z, z*z*z
        ''')

    I_g = np.array(
                 [1, 1, 1, 1, 1, 1, 
                  2, 2, 2, 
                  3, 3, 3, 
                  4, 4, 4, 4, 
                  5 ,5, 5, 
                  6, 6, 
                  7, 7, 
                  8, 8, 8, 
                  9, 9]) - 1
    J_g = np.array(
                 [1, 4, 6, 11, 13, 15, 
                  3, 8, 10, 
                  5, 12, 14, 
                  1, 6, 11, 13, 
                  2, 7, 9, 
                  3, 8, 
                  5, 12, 
                  1, 4, 11, 
                  2, 7]) - 1
    V_g = Matrix(
                 [3/8, 3*sqrt(105)/140, -3*sqrt(105)/35, 3/8, -3*sqrt(105)/35, 1, 
                 -3*sqrt(70)/28, -3*sqrt(14)/28, sqrt(70)/7, 
                 -3*sqrt(14)/28, -3*sqrt(70)/28, sqrt(70)/7, 
                 -sqrt(5)/4, 3*sqrt(21)/14, sqrt(5)/4, -3*sqrt(21)/14, 
                 -sqrt(35)/14, -sqrt(35)/14, 3*sqrt(7)/7, 
                 sqrt(10)/4, -3*sqrt(2)/4, 
                 3*sqrt(2)/4, -sqrt(10)/4, 
                 sqrt(35)/8, -3*sqrt(3)/4, sqrt(35)/8, 
                 sqrt(5)/2, -sqrt(5)/2])
    sz_g = (9, 15)
    U_g = re.sub('\n', ' ', '''
        x*x*x*x, sqrt(7)*x*x*x*y, sqrt(7)*x*x*x*z, 
        sqrt(35/3)*x*x*y*y, sqrt(35)*x*x*y*z, sqrt(35/3)*x*x*z*z, 
        sqrt(7)*x*y*y*y, sqrt(35)*x*y*y*z, sqrt(35)*x*y*z*z, 
        sqrt(7)*x*z*z*z, y*y*y*y, sqrt(7)*y*y*y*z, 
        sqrt(35/3)*y*y*z*z, sqrt(7)*y*z*z*z, z*z*z*z
        ''')

    I_h = np.array(
                 [1, 1, 1, 1, 1, 1, 
                  2, 2, 2, 2, 2, 2, 
                  3, 3, 3, 3, 3, 3, 
                  4, 4, 4, 4, 
                  5 ,5, 5, 
                  6, 6, 6, 6, 6, 
                  7, 7, 7, 7, 7, 
                  8, 8, 8, 
                  9, 9, 
                  10, 10, 10, 
                  11, 11, 11]) - 1
    J_h = np.array(
                 [3, 8, 10, 17, 19, 21, 
                  1, 4, 6, 11, 13, 15, 
                  2, 7, 9, 16, 18, 20, 
                  3, 10, 17, 19, 
                  5, 12, 14, 
                  1, 4, 6, 11, 13, 
                  2, 7, 9, 16, 18, 
                  3, 8, 17, 
                  5, 12, 
                  1, 4, 11, 
                  2, 7, 16]) - 1
    V_h = Matrix(        
                 [5/8, sqrt(105)/28, -5/sqrt(21), 5/8, -5/sqrt(21), 1, 
                  sqrt(15)/8, sqrt(35)/28, -3*sqrt(35)/14, sqrt(15)/24, -3*sqrt(7)/14, sqrt(15)/3, 
                  sqrt(15)/24, sqrt(35)/28, -3*sqrt(7)/14, sqrt(15)/8, -3*sqrt(35)/14, sqrt(15)/3, 
                  -sqrt(105)/12, sqrt(5)/2, sqrt(105)/12, -sqrt(5)/2, 
                  -sqrt(15)/6, -sqrt(15)/6, sqrt(15)/3, 
                  -sqrt(70)/16, sqrt(30)/24, sqrt(30)/6, sqrt(70)/16, -sqrt(6)/2, 
                  -sqrt(70)/16, -sqrt(30)/24, sqrt(6)/2, sqrt(70)/16, -sqrt(30)/6, 
                  sqrt(35)/8, -3*sqrt(3)/4, sqrt(35)/8, 
                  sqrt(5)/2, -sqrt(5)/2, 
                  3*sqrt(14)/16, -5*sqrt(6)/8, 5*sqrt(14)/16, 
                  5*sqrt(14)/16, -5*sqrt(6)/8, 3*sqrt(14)/16])
    sz_h = (11, 21)
    U_h = re.sub('\n', ' ', '''
        x*x*x*x*x, 3*x*x*x*x*y, 3*x*x*x*x*z, 
        3*sqrt(7/3)*x*x*x*y*y, 3*sqrt(7)*x*x*x*y*z, 3*sqrt(7/3)*x*x*x*z*z, 
        3*sqrt(7/3)*x*x*y*y*y, 3*sqrt(35/3)*x*x*y*y*z, 3*sqrt(35/3)*x*x*y*z*z, 
        3*sqrt(7/3)*x*x*z*z*z, 3*x*y*y*y*y, 3*sqrt(7)*x*y*y*y*z, 
        3*sqrt(35/3)*x*y*y*z*z, 3*sqrt(7)*x*y*z*z*z, 3*x*z*z*z*z, 
        y*y*y*y*y, 3*y*y*y*y*z, 3*sqrt(7/3)*y*y*y*z*z, 
        3*sqrt(7/3)*y*y*z*z*z, 3*y*z*z*z*z, z*z*z*z*z
        ''')
    
    I = eval('I_' + ang.lower())
    J = eval('J_' + ang.lower())
    V = eval('V_' + ang.lower())
    sz = eval('sz_' + ang.lower())
    U = eval('U_' + ang.lower())

    return I, J, V, sz, U


def solid_harm_sm(ang, trans_dict=None, order='pyscf'):
    from symbolic_math import sympy_coo_matrix, to_sympy_mat, transform_expr    
    I, J, V, sz, U_str = find_IJVszU(ang)
    M = sympy_coo_matrix(I, J, V, sz)
    U = to_sympy_mat(U_str)
    if trans_dict is not None:
        U = transform_expr(U, trans_dict)
    MxU = to_pyscf_sh_order(M * U) if order=='pyscf' else M * U
    return MxU
def to_pyscf_sh_order(V):
    from sympy import Matrix
    V = Matrix(V)
    assert 1 in V.shape
    n = V.shape[0] if V.shape[0] > V.shape[1] else V.shape[1]
    if n in [5, 7, 9, 11]:
        idces = [s for s in range(2,n,2)][::-1] + [0] + [c for c in range(1,n,2)]
        V = Matrix([V[idces[i]] for i in range(n)])
    return V
    

# def solid_harm_poly(angs, trans_dict=None, order='pyscf'):
#     from sympy import exp, sqrt
#     x, y, z = sm.symbols('x, y, z')
#     x_ori, y_ori, z_ori = sm.symbols('x_ori, y_ori, z_ori')
#     harm_list = []
#     t0 = time.time()
#     for ang in angs:
#         V = solid_harm_sm(ang, trans_dict, order)
#         harm_list += [sm.simplify(harm.subs({x:x+x_ori, y:y+y_ori, z:z+z_ori}).as_poly(x, y, z)) for harm in V]    
#     t1 = time.time()
#     monoms = [harm_list[i].monoms() for i in range(len(V))]
#     coeffs = [harm_list[i].coeffs() for i in range(len(V))]
#     t2 = time.time()
#     print('t0-1:', t1-t0)
#     print('t1-2:', t2-t1)
#     return monoms, coeffs

def solid_harm_poly_sm(angs, trans_dict=None, order='pyscf'):
    from sympy import exp, sqrt
    from symbolic_math import transform_expr_pure_sm
    x, y, z = sm.symbols('x, y, z')
    poly_list = []
    t0 = time.time()
    for ang in angs:
        V = solid_harm_sm(ang, None, order)
        V, sybs = transform_expr_pure_sm(V, trans_dict).values()
        for syb in sybs: sm.var(syb)
        poly_list += [sm.simplify(harm.as_poly(x, y, z)) for harm in V]
        # print(harm_list)
    t1 = time.time()
    print('time finding poly:', t1-t0)
    return poly_list
def solid_harm_poly_LS(highest_ang, trans_dict, folder='shs', 
                           to_str=True, to_torch=True, flatten=True,
                           load_sh=True, save_sh=True):
    import os.path as osp
    this_file = osp.abspath(__file__)
    this_dir = osp.dirname(this_file)
    sh_dir = osp.join(this_dir, folder)
    # trans_dict
    if 'translation' in trans_dict:
        t = 't' if trans_dict['translation'] else ''
    else: t = ''
    if 'rotation' in trans_dict:
        r = 'r' if trans_dict['rotation'] else ''
    else: r = ''
    if 'derivative' in trans_dict:
        d = trans_dict['derivative']
    else: d = '0'
    # angs
    ang_dict = {1:'s', 3:'p', 5:'d', 7:'f', 9:'g', 11:'h'}
    if type(highest_ang) == int: highest_ang = ang_dict[highest_ang]
    highest_ang = highest_ang.lower()
    if not to_str: front_factor_type = 'sympy'
    else: front_factor_type = 'torch' if to_torch else 'numpy'
    import os.path as osp
    import pickle
    if load_sh:
        for ang in list(ang_dict.keys())[list(ang_dict.values()).index(highest_ang):]:
            fp = osp.join(sh_dir, 'sh_2ang_%s_%s_%s_d_%s_%s.pkl'%(highest_ang, t, r, d, front_factor_type))
            if osp.exists(fp):
                with open(fp, 'rb') as f:
                    sh = pickle.load(f)
                return sh
    # if file not found, calculate
    sh = {}
    from symbolic_math import poly_to_list_of_dict
    for ang in list(ang_dict.keys())[:list(ang_dict.values()).index(highest_ang)+1]:
        poly_list = solid_harm_poly_sm(angs = [ang_dict[ang]], trans_dict=trans_dict, order='pyscf')
        poly_list_of_dict = poly_to_list_of_dict(poly_list, to_str=to_str, to_torch=to_torch)
        if flatten: poly_list_of_dict = flatten_poly(poly_list_of_dict)
        sh[ang] = poly_list_of_dict
    if save_sh:
        fp = fp = osp.join(sh_dir, 'sh_2ang_%s_%s_%s_d_%s_%s.pkl'%(highest_ang, t, r, d, front_factor_type))
        with open(fp, 'wb') as f:
            pickle.dump(sh, f)
    return sh

# def sh_on_grid():
#     ...


# def solid_harm_flat_torch(highest_ang, r_ref):
#     # evaluate with tensor of x,y,z ref, no rotation, no derivatives
#     poly = solid_harm_poly_n_save(highest_ang, trans_dict={'translation':True}, folder='shs', to_str=True, to_torch=True, load_sh=True, save_sh=True)
#     poly = [d for key in poly for d in poly[key]]  ?
#     from symbolic_math import flatten_poly
#     poly = flatten_poly(poly)
#     x_ref, y_ref, z_ref = r_ref.split(1, dim=-1)
#     x_ref, y_ref, z_ref = x_ref.squeeze(-1), y_ref.squeeze(-1),z_ref.squeeze(-1)
#     #
#     import torch
#     orders = torch.tensor(poly['orders'])
#     n_terms = len(poly['poly_indces'])
#     r_shape = r_ref.shape[:-1]
#     coeffs = torch.empty((n_terms,)+r_shape, dtype=torch.float32)
#     for i_poly in range(n_terms):
#         print(coeffs[i_poly].shape)
#         print('t',poly['coeffs'][i_poly])
#         coeffs[i_poly] = eval(poly['coeffs'][i_poly])
#     idx = torch.tensor(poly['poly_indces'])
#     return {'orders':orders, 'coeffs':coeffs, 'poly_indces':idx, 'n_polyes':poly['n_polyes']}
    

def solid_harm(ang, x, y, z, trans_dict=None, deriv=0, package='numpy', device='cpu'):
    # import numpy as np
    # import torch
    # import sympy as sm
    # from symbolic_math import sympy_coo_matrix, to_sympy_mat, eval_sympy_mat, transform_expr

    # I, J, V, sz, U_str = find_IJVszU(ang)
    # M = sympy_coo_matrix(I, J, V, sz)
    # U = to_sympy_mat(U_str)
    # if trans_dict is not None:
    #     U = transform_expr(U, trans_dict)
    
    # if deriv in ['x', 'y', 'z']:
    #     U = U.applyfunc(lambda f: sm.diff(f,deriv))
    # elif deriv == 1 or deriv == (0, 1):
    #     dx_U = U.applyfunc(lambda f: sm.diff(f,'x'))
    #     dy_U = U.applyfunc(lambda f: sm.diff(f,'y'))
    #     dz_U = U.applyfunc(lambda f: sm.diff(f,'z'))
    #     if deriv == 1: U = dx_U.row_join(dy_U).row_join(dz_U)
    #     elif deriv == (0, 1): U = U.row_join(dx_U).row_join(dy_U).row_join(dz_U)
    # return eval_sympy_mat(M*U, x, y, z, deriv=0, package=package, device=device)
    from symbolic_math import eval_sympy_mat
    MxU_mat = solid_harm_sm(ang, trans_dict=trans_dict)
    MxU = eval_sympy_mat(MxU_mat, x, y, z, deriv=deriv, package=package, device=device)
    # print(MxU)
    # if package == 'numpy': MxU = np.squeeze(MxU, 1)
    # if package == 'torch': MxU = MxU.squeeze(1)
    return MxU


def test():
    import numpy as np
    import torch
    import sympy as sm
    # from symbolic_math import poly_monoms_coeffs

    # poly_list = solid_harm_poly_sm(['d'], {'rotation':True, 'translation':True})

    # t0 = time.time()
    # m, co = poly_monoms_coeffs(poly_list, to_torch=True)
    # print(m)
    # x_ref, y_ref, z_ref, v_x, v_y, v_z, omega = [torch.randn(1000, 1000, device='cuda')]*7
    # device = 'cuda'
    # # from torch import sin, cos, exp, sqrt
    # t1 = time.time()
    # for i in co:
    #     for j in i:
    #         print(eval(j))
    # t2 = time.time()
    # print('time_str:',t1-t0)
    # print('time_eval:',t2-t1)
    # print([[expr.subs({'x_ref':0.0, 'y_ref':0.0, 'z_ref':0.0, 'omega':0.0, 'v_x':0.0, 'v_y':0.0, 'v_z':0.0}) for expr in l] for l in co])
    
    sh = solid_harm_poly_LS('p', {'translation':True}, 'shs')
    print(sh)



    # for ang in ['d']:
    # for ang in ['s', 'p', 'd', 'f', 'g', 'h']:
        # print('ang = ', ang)
        # x = np.random.randn(10)
        # y = np.random.randn(10)
        # z = np.random.randn(10)
        # print(solid_harm(ang, x, y, z).shape )
        # x = np.array([1])
        # y = np.array([1])
        # z = np.array([1])
        # print(solid_harm(ang, x, y, z, deriv='x') )
        # x = np.random.randn(1000, 1000)
        # y = np.random.randn(1000, 1000)
        # z = np.random.randn(1000, 1000)
        # print(solid_harm(ang, x, y, z).shape )       

        # device = 'cuda'
        # x = torch.randn(10, device=device)
        # y = torch.randn(10, device=device)
        # z = torch.randn(10, device=device)
        # print(solid_harm(ang, x, y, z, package='torch', device=device).shape )
        # x = torch.tensor([1], device=device)
        # y = torch.tensor([1], device=device)
        # z = torch.tensor([1], device=device)
        # print(solid_harm(ang, x, y, z, deriv=(0,1), package='torch', device=device).shape )
        # x = torch.randn((1000, 1000), device=device)
        # y = torch.randn((1000, 1000), device=device)
        # z = torch.randn((1000, 1000), device=device)
        # print(solid_harm(ang, x, y, z, package='torch', device=device).shape )
        
if __name__ == '__main__':
    test()

