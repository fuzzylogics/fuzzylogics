import sympy as sm
import numpy as np
import torch
import time

#_generate_matrix______________________________________________
def sympy_coo_matrix(I, J, V, size):
    # sort I, J, V such that I is monotonic increasing
    from sympy import Matrix
    I, J, V = list(I), list(J), Matrix(V)
    I_idx = sorted(range(len(I)),key=I.__getitem__)
    I = [int(I[i]) for i in I_idx]
    J = [int(J[i]) for i in I_idx]
    V = Matrix([V[i] for i in I_idx])
    # the row cut index for csr
    I_cuts = [0]
    x = 0
    for i in range(len(I)):
        if x < I[i]:
            for n in range(I[i]-x):
                I_cuts += [i]
                x = I[i]
    I_cuts += [len(I)]
    # csr in sympy
    from sympy.matrices.sparsetools import _csrtodok
    M = _csrtodok([V, J, I_cuts, size])
    return M
#______________________________________________________________


#_Matrix_format_convertions:___________________________________
def sympy_mat_to_str(M):
    mat_str = "["
    for i in range(M.shape[0]):
        row = "["
        for j in range(M.shape[1]):
            row += str(M[i,j])
            row += ', '
        row += ']'
        mat_str += row
        mat_str += ', '
    mat_str += ']'
    return mat_str

def sympy_mat_to_list_of_str(M):
    mat_list = []
    for i in range(M.shape[0]):
        row = []
        for j in range(M.shape[1]):
            row += [str(M[i,j])]
        mat_list += [row]
    return mat_list

def to_sympy_mat(mat):
    if type(mat) == list:
        M = sm.Matrix(mat)
    elif type(mat) == str:
        from sympy.parsing.sympy_parser import parse_expr
        # vector will become n*1 matrix
        parsed_mat = parse_expr(mat)
        if type(parsed_mat) not in [list, tuple]:
            parsed_mat = [parsed_mat]
        M = sm.Matrix(parsed_mat)
    return M
#______________________________________________________________


#_evaluate_matrix______________________________________________
def eval_mat_list(mat_list, x, y, z, package='numpy', device='cpu'):
    t0 = time.time()
    shape = (len(mat_list), len(mat_list[0])) + x.shape
    if package == 'numpy':
        res = np.empty(shape, dtype=float)
        exp, sqrt = np.exp, np.sqrt
    elif package == 'torch':
        res = torch.empty(shape, dtype=torch.float32, device=device)
        exp, sqrt = torch.exp, torch.sqrt
        import re
        def add_tensor(s):
            if not re.findall('[a-z]|[A-Z]', s):
            # if 'x' not in s and 'y' not in s and 'z' not in s:
                new_s = 'torch.tensor(' + s + ', dtype=torch.float32, device=device)'
            else: new_s = s
            return new_s
    # evaluate element by element
    t1 = time.time()
    t_expr, t_eval = 0.0, 0.0
    for i in range(len(mat_list)):
        for j in range(len(mat_list[0])):

            tt0 = time.time()

            if package == 'torch': # number must be tensor for torch arithmetics
                str2eval = re.sub('(exp|sqrt)\((.*?)\)', lambda x:x.group(1)+'('+add_tensor(x.group(2))+')', mat_list[i][j])
            else: str2eval = mat_list[i][j]

            tt1 = time.time()

            res[i, j] = eval(str2eval)

            tt2 = time.time()

            t_expr += (tt1 - tt0)
            t_eval += (tt2 - tt1)
    # print("time for preparation: ", t1 - t0)
    # print("time for generating expressions: ", t_expr)
    # print("time for evaluating at x, y, z: ", t_eval)
    return res

def torchify_str(expr_str):
    import re
    def add_tensor(s):
        if 'x' not in s and 'y' not in s and 'z' not in s:
            new_s = 'torch.tensor(' + s + ', dtype=torch.float32, device=device)'
        else: new_s = s
        return new_s
    # cannot handle nested exp(sqrt(...))
    str2eval = re.sub('(exp|sqrt|sin|cos)\((.*?)\)', lambda x:'torch.'+x.group(1)+'('+add_tensor(x.group(2))+')', expr_str)
    return str2eval


def eval_sympy_mat(M, x, y, z, deriv=0, package='numpy', device='cpu'):
    # deriv is the order of (elemen-by-element) derivative
    import sympy as sm
    t0 = time.time()
    if type(deriv) == int: deriv = (deriv,)
    if deriv in ['x', 'y', 'z']:
        Mp = M.applyfunc(lambda f: sm.diff(f,deriv))
    else:
        Mp = sm.Matrix()
        if 0 in deriv:
            Mp = Mp.row_join(M)
        if 1 in deriv:
            assert 1 in M.shape, "can only handle vectors for derivative ..."
            if M.shape[0] == 1:
                M, Mp = M.T, Mp.T
            dx_M = M.applyfunc(lambda f: sm.diff(f,'x'))
            dy_M = M.applyfunc(lambda f: sm.diff(f,'y'))
            dz_M = M.applyfunc(lambda f: sm.diff(f,'z'))
            Mp = Mp.row_join(dx_M).row_join(dy_M).row_join(dz_M)
    # convert to list then evaluate
    t1 = time.time()
    mat_list = sympy_mat_to_list_of_str(Mp)
    t2 = time.time()
    res = eval_mat_list(mat_list, x, y, z, package=package, device=device)
    t3 = time.time()

    # print("time for derivation: ", t1 - t0)
    # print("time for mat to str: ", t2 - t1)
    # print("time for evaluation: ", t3 - t2)

    ## remove one dim from vector M
    # if squeezeM:
    #     if len(mat_list[0]) == 1:
    #         if package == 'numpy': res = np.squeeze(res, axis=1)
    #         elif package == 'torch': res = res.squeeze(1)
    #     elif len(mat_list) == 1: # doesn't remove another dim even for 1x1 mat
    #         if package == 'numpy': res = np.squeeze(res, axis=0)
    #         elif package == 'torch': res = res.squeeze(0)
    return res
#______________________________________________________________


# polynomials__________________________________________________
def poly_to_list_of_dict(poly_list, to_str=True, to_torch=False):
    # monoms = [poly.monoms() for poly in poly_list]
    # coeffs = [poly.coeffs() for poly in poly_list]
    # if to_str:
    #     coeffs = [[str(co) for co in co_s] for co_s in coeffs]
    #     if to_torch:
    #         coeffs = [[torchify_str(co) for co in co_s] for co_s in coeffs]
    # return monoms, coeffs
    if to_str:
        func = lambda x: str(x)
        if to_torch: func = lambda x: torchify_str(str(x))
    else: func = lambda x: x
    poly_list_of_dict = [dict(zip(poly.monoms(), [func(ele) for ele in poly.coeffs()])) for poly in poly_list]
    return poly_list_of_dict

def multiply_polyes_sm(p1, p2):
    # p1, p2 are list_of_dicts
    return [{x: A.get(x, 1)*(B.get(x, 1)) for x in set(A).union(B)} for B in p2 for A in p1]
def multiply_polyes_str(p1, p2):
    # p1, p2 are list_of_dicts
    return [{x: '('+A.get(x, 1)+')*('+B.get(x, 1)+')' for x in set(A).union(B)} for B in p2 for A in p1]

def flatten_poly(poly):
    if type(poly[0]) == list:
        poly = poly_to_list_of_dict(poly)
    i_poly = 0
    idx, orders, coeffs = [], [], []
    for poly_dict in poly:
        idx += [i_poly]*len(poly_dict)
        orders += list(poly_dict.keys())
        coeffs += list(poly_dict.values())
        i_poly += 1
    return {'orders':orders, 'coeffs':coeffs, 'poly_indces':idx, 'n_polyes':i_poly}

def eval_poly_r_ref_torch(poly, r_ref):
    # evaluate with tensor of x,y,z ref, no rotation, no derivatives
    poly = flatten_poly(poly)
    x_ref, y_ref, z_ref = r_ref.split(1, dim=-1)
    x_ref, y_ref, z_ref = x_ref.squeeze(-1), y_ref.squeeze(-1),z_ref.squeeze(-1)
    #
    import torch
    orders = torch.tensor(poly['orders'])
    n_terms = len(poly['poly_indces'])
    r_shape = r_ref.shape[:-1]
    coeffs = torch.empty((n_terms,)+r_shape, dtype=torch.float32)
    for i_poly in range(n_terms):
        print(coeffs[i_poly].shape)
        print('t',poly['coeffs'][i_poly])
        coeffs[i_poly] = eval(poly['coeffs'][i_poly])
    idx = torch.tensor(poly['poly_indces'])
    return {'orders':orders, 'coeffs':coeffs, 'poly_indces':idx, 'n_polyes':poly['n_polyes']}

#______________________________________________________________    



#_transformations____________________________________________
def transform_expr(expr, trans_dict): # expr can also be a sympy matrix
    x, y, z = sm.symbols('x, y, z')
    xo, yo, zo = sm.symbols('xo, yo, zo')
    # xn, yn, zn = sm.symbols('xn, yn, zn')

    if 'rotation' in trans_dict and trans_dict['rotation'] is not None:
        R = sm.Matrix(trans_dict['rotation'])
        R = R.inv()
        V = sm.Matrix([x, y, z])
        xn, yn, zn = R * V
        # sympy will mess up notations if substituded directly
        expr = expr.subs({x:xo, y:yo, z:zo})
        expr = expr.subs({xo:xn, yo:yn, zo:zn})
        
    if 'translation' in trans_dict and trans_dict['translation'] is not None:
        T = trans_dict['translation']
        xp, yp, zp = x-T[0], y-T[1], z-T[2]
        expr = expr.subs({x:xp, y:yp, z:zp})

    if 'derivative' in trans_dict and trans_dict['derivative'] is not None:
        if ('x' or 'y' or 'z') in trans_dict['derivative']:
            cdx = trans_dict['derivative'].count(x)
            cdy = trans_dict['derivative'].count(y)
            cdz = trans_dict['derivative'].count(z)
            expr = expr.diff('x', cdx, 'y', cdy, 'z', cdz)
        if trans_dict['derivative'].strip().lower() == 'laplace':
            expr = expr.diff('x', 2) + expr.diff('y', 2) + expr.diff('z', 2)           
    return expr


def transform_expr_pure_sm(expr, trans_dict):
    # the contents of rotation and translation in trans_dict are ignored, expr contains symbols redefined
    if trans_dict is None: return {'expr':expr, 'listOfSymbolStrings':[]}
    x, y, z = sm.symbols('x, y, z')
    x_o, y_o, z_o = sm.symbols('x_o, y_o, z_o')
    # sybs = ['x', 'y', 'z']
    sybs = []
    if 'rotation' in trans_dict and trans_dict['rotation'] is not None and trans_dict['rotation']!=False:
        R, R_sybs = rotation_mat_sm().values()
        V = sm.Matrix([x, y, z])
        x_n, y_n, z_n = R * V
        # sympy will mess up notations if substituded directly
        expr = expr.subs({x:x_o, y:y_o, z:z_o})
        expr = expr.subs({x_o:x_n, y_o:y_n, z_o:z_n})
        sybs += R_sybs       
    if 'translation' in trans_dict and trans_dict['translation'] is not None and trans_dict['translation']!=False:
        x_ref, y_ref, z_ref = sm.symbols('x_ref, y_ref, z_ref')
        x_n, y_n, z_n = x-x_ref, y-y_ref, z-z_ref
        expr = expr.subs({x:x_o, y:y_o, z:z_o})
        expr = expr.subs({x_o:x_n, y_o:y_n, z_o:z_n})
        sybs += ['x_ref', 'y_ref', 'z_ref']
    if 'derivative' in trans_dict and trans_dict['derivative'] is not None and trans_dict['derivative']!=False:
        if ('x' or 'y' or 'z') in trans_dict['derivative']:
            cdx = trans_dict['derivative'].count('x')
            cdy = trans_dict['derivative'].count('y')
            cdz = trans_dict['derivative'].count('z')
            expr = expr.diff('x', cdx, 'y', cdy, 'z', cdz)
        if trans_dict['derivative'].strip().lower() == 'laplace':
            expr = expr.diff('x', 2) + expr.diff('y', 2) + expr.diff('z', 2)           
    return {'expr':expr, 'listOfSymbolStrings':sybs}
def rotation_mat_sm():
    from sympy.vector import CoordSys3D
    from sympy import symbols
    # rotation angle: omega, rotation vector: v_x, v_y, v_z (normalized)
    omega = symbols('omega')
    v_x, v_y, v_z = symbols('v_x, v_y, v_z')
    # coordinate systems old: xyz, new: XYZ
    xyz = CoordSys3D('xyz')
    from sympy.vector import AxisOrienter
    orienter = AxisOrienter(omega, v_x*xyz.i + v_y*xyz.j + v_z*xyz.k)
    XYZ = xyz.orient_new('XYZ', (orienter, ))
    rot_mat_sm = xyz.rotation_matrix(XYZ).subs(v_x**2+v_y**2+v_z**2, 1)
    return {'mat':rot_mat_sm, 'listOfSymbolStrings':[str(x) for x in ['omega', 'v_x', 'v_y', 'v_z']]}

#______________________________________________________________


def test():
    print(to_sympy_mat('1.0'))
    print(to_sympy_mat('x*x, sqrt(3)*x*y, sqrt(3)*x*z, y*y, sqrt(3)*y*z, z*z'))
    print(to_sympy_mat('x, y, z'))
if __name__ == '__main__':
    test()
