# basis
import numpy as np
from numpy import sqrt
import torch
from scipy.sparse import coo_matrix
import sympy as sm
from symbolic_math import *

def find_solid_harm_func_by_ang(ang):
    func_switcher = {
        's': solid_harm_s,
        'S': solid_harm_s,
        'p': solid_harm_p,
        'P': solid_harm_p,
        'd': solid_harm_d,
        'D': solid_harm_d,
        'f': solid_harm_f,
        'F': solid_harm_f,
        'g': solid_harm_g,
        'G': solid_harm_g,
        'h': solid_harm_h,
        'H': solid_harm_h
    }
    return func_switcher.get(ang, lambda x,y,z:"invalid solid harmonic.")

def solid_harm_s(x, y, z, package='scipy', device='cpu'):
    if package == 'scipy':
        return np.expand_dims(np.ones_like(x), -1)
    elif package == 'torch':
        return torch.ones_like(x, device=device).unsqueeze(-1)

def solid_harm_p(x, y, z, package='scipy', device='cpu'):
    if package == 'scipy':
        return np.stack((x, y, z), -1)
    elif package == 'torch':
        return torch.stack((x, y, z), -1).to(device)


def solid_harm_d(x, y, z, package='scipy', device='cpu'):
    I = np.array([1, 1, 1, 2, 3, 4, 4, 5])-1
    J = np.array([1, 4, 6, 3, 5, 1, 4, 2])-1
    V = np.array([-1/2, -1/2, 1, 1, 1, sqrt(3)/2, -sqrt(3)/2, 1])
    U_str = 'x*x, sqrt(3)*x*y, sqrt(3)*x*z, y*y, sqrt(3)*y*z, z*z'
    if package == 'scipy':
        M = coo_matrix( (V,(I,J)), shape=(5,6) ).toarray()
        U = np.stack((eval(U_str)), axis=-1)
        return np.matmul(U, M.T)
    elif package == 'torch':
        M = torch.sparse_coo_tensor( np.stack((I,J),axis=0), V, (5,6) ).float().to_dense().to(device)
        U = torch.stack((eval(U_str)), dim=-1).to(device)
        return torch.matmul(U, M.T)


# def solid_harm_d(x, y, z, package='numpy', device='cpu', deriv=None):
#     I = np.array([1, 1, 1, 2, 3, 4, 4, 5])-1
#     J = np.array([1, 4, 6, 3, 5, 1, 4, 2])-1
#     V = np.array([-1/2, -1/2, 1, 1, 1, sqrt(3)/2, -sqrt(3)/2, 1])
#     M = sympy_coo_matrix(I, J, V, (5,6))
#     U = to_sympy_mat('x*x, sqrt(3)*x*y, sqrt(3)*x*z, y*y, sqrt(3)*y*z, z*z')
#     if deriv in ['x', 'y', 'z']:
#         U = U.applyfunc(lambda f: sm.diff(f,deriv))
#     return eval_sympy_mat(M*U, x, y, z, package=package, device=device)
  
        
def solid_harm_f(x, y, z, package='scipy', device='cpu'):
    I = np.array(
        [1, 1, 1, 
         2, 2, 2, 
         3, 3, 3, 
         4, 4, 
         5, 
         6, 6, 
         7, 7]) - 1
    J = np.array(
        [3, 8, 10, 
         1, 4, 6, 
         2, 7, 9, 
         3, 8, 
         5, 
         1, 4, 
         2, 7]) - 1
    V = np.array(
        [-3*sqrt(5)/10, -3*sqrt(5)/10, 1,
         -sqrt(6)/4, -sqrt(30)/20, sqrt(30)/5,
         -sqrt(30)/20, -sqrt(6)/4, sqrt(30)/5,
         sqrt(3)/2, -sqrt(3)/2,
         1,
         sqrt(10)/4, -3*sqrt(2)/4,
         3*sqrt(2)/4, -sqrt(10)/4])
    if package == 'scipy':
        M = coo_matrix( (V,(I,J)), shape=(7,10) ).toarray()
        U = np.stack(
                     (x*x*x, sqrt(5)*x*x*y, sqrt(5)*x*x*z, sqrt(5)*x*y*y, sqrt(15)*x*y*z, 
                      sqrt(5)*x*z*z, y*y*y, sqrt(5)*y*y*z, sqrt(5)*y*z*z, z*z*z),
                     axis=-1)
        return np.matmul(U, M.T)
    elif package == 'torch':
        M = torch.sparse_coo_tensor( np.stack((I,J),axis=0), V, (7,10) ).float().to_dense().to(device)
        U = torch.stack(
                        (x*x*x, sqrt(5)*x*x*y, sqrt(5)*x*x*z, sqrt(5)*x*y*y, sqrt(15)*x*y*z,
                         sqrt(5)*x*z*z, y*y*y, sqrt(5)*y*y*z, sqrt(5)*y*z*z, z*z*z),
                        dim=-1).to(device)
        return torch.matmul(U, M.T)       

def solid_harm_g(x, y, z, package='scipy', device='cpu'):
    I = np.array(
                 [1, 1, 1, 1, 1, 1, 
                  2, 2, 2, 
                  3, 3, 3, 
                  4, 4, 4, 4, 
                  5 ,5, 5, 
                  6, 6, 
                  7, 7, 
                  8, 8, 8, 
                  9, 9]) - 1
    J = np.array(
                 [1, 4, 6, 11, 13, 15, 
                  3, 8, 10, 
                  5, 12, 14, 
                  1, 6, 11, 13, 
                  2, 7, 9, 
                  3, 8, 
                  5, 12, 
                  1, 4, 11, 
                  2, 7]) - 1
    V = np.array(
                 [3/8, 3*sqrt(105)/140, -3*sqrt(105)/35, 3/8, -3*sqrt(105)/35, 1, 
                 -3*sqrt(70)/28, -3*sqrt(14)/28, sqrt(70)/7, 
                 -3*sqrt(14)/28, -3*sqrt(70)/28, sqrt(70)/7, 
                 -sqrt(5)/4, 3*sqrt(21)/14, sqrt(5)/4, -3*sqrt(21)/14, 
                 -sqrt(35)/14, -sqrt(35)/14, 3*sqrt(7)/7, 
                 sqrt(10)/4, -3*sqrt(2)/4, 
                 3*sqrt(2)/4, -sqrt(10)/4, 
                 sqrt(35)/8, -3*sqrt(3)/4, sqrt(35)/8, 
                 sqrt(5)/2, -sqrt(5)/2])     
    if package == 'scipy':
        M = coo_matrix( (V,(I,J)), shape=(9,15) ).toarray()
        U = np.stack(
                     (x*x*x*x, sqrt(7)*x*x*x*y, sqrt(7)*x*x*x*z, sqrt(35/3)*x*x*y*y, sqrt(35)*x*x*y*z, sqrt(35/3)*x*x*z*z, 
                      sqrt(7)*x*y*y*y, sqrt(35)*x*y*y*z, sqrt(35)*x*y*z*z, sqrt(7)*x*z*z*z, 
                      y*y*y*y, sqrt(7)*y*y*y*z, sqrt(35/3)*y*y*z*z, sqrt(7)*y*z*z*z, z*z*z*z),
                     axis=-1)
        return np.matmul(U, M.T)
    elif package == 'torch':
        M = torch.sparse_coo_tensor( np.stack((I,J),axis=0), V, (9,15) ).float().to_dense().to(device)
        U = torch.stack(
                        (x*x*x*x, sqrt(7)*x*x*x*y, sqrt(7)*x*x*x*z, sqrt(35/3)*x*x*y*y, sqrt(35)*x*x*y*z, sqrt(35/3)*x*x*z*z, 
                         sqrt(7)*x*y*y*y, sqrt(35)*x*y*y*z, sqrt(35)*x*y*z*z, sqrt(7)*x*z*z*z, 
                         y*y*y*y, sqrt(7)*y*y*y*z, sqrt(35/3)*y*y*z*z, sqrt(7)*y*z*z*z, z*z*z*z),
                        dim=-1).to(device)
        return torch.matmul(U, M.T)
    

def solid_harm_h(x, y, z, package='scipy', device='cpu'):
    I = np.array(
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
    J = np.array(
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
    V = np.array(        
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
    if package == 'scipy':
        M = coo_matrix( (V,(I,J)), shape=(11,21) ).toarray()
        U = np.stack(
                     (x*x*x*x*x, 3*x*x*x*x*y, 3*x*x*x*x*z, 3*sqrt(7/3)*x*x*x*y*y, 3*sqrt(7)*x*x*x*y*z, 3*sqrt(7/3)*x*x*x*z*z, 
                      3*sqrt(7/3)*x*x*y*y*y, 3*sqrt(35/3)*x*x*y*y*z, 3*sqrt(35/3)*x*x*y*z*z, 3*sqrt(7/3)*x*x*z*z*z, 
                      3*x*y*y*y*y, 3*sqrt(7)*x*y*y*y*z, 3*sqrt(35/3)*x*y*y*z*z, 3*sqrt(7)*x*y*z*z*z, 3*x*z*z*z*z, 
                      y*y*y*y*y, 3*y*y*y*y*z, 3*sqrt(7/3)*y*y*y*z*z, 3*sqrt(7/3)*y*y*z*z*z, 3*y*z*z*z*z, z*z*z*z*z), 
                     axis=-1)
        return np.matmul(U, M.T)
    elif package == 'torch':
        M = torch.sparse_coo_tensor( np.stack((I,J),axis=0), V, (11,21) ).float().to_dense().to(device)
        U = torch.stack(
                        (x*x*x*x*x, 3*x*x*x*x*y, 3*x*x*x*x*z, 3*sqrt(7/3)*x*x*x*y*y, 3*sqrt(7)*x*x*x*y*z, 3*sqrt(7/3)*x*x*x*z*z, 
                         3*sqrt(7/3)*x*x*y*y*y, 3*sqrt(35/3)*x*x*y*y*z, 3*sqrt(35/3)*x*x*y*z*z, 3*sqrt(7/3)*x*x*z*z*z, 
                         3*x*y*y*y*y, 3*sqrt(7)*x*y*y*y*z, 3*sqrt(35/3)*x*y*y*z*z, 3*sqrt(7)*x*y*z*z*z, 3*x*z*z*z*z, 
                         y*y*y*y*y, 3*y*y*y*y*z, 3*sqrt(7/3)*y*y*y*z*z, 3*sqrt(7/3)*y*y*z*z*z, 3*y*z*z*z*z, z*z*z*z*z),
                        dim=-1).to(device)
        return torch.matmul(U, M.T)


def test():
    for ang in ['s', 'p', 'd', 'f', 'g', 'h']:
        x = np.random.randn(10)
        y = np.random.randn(10)
        z = np.random.randn(10)
        print(find_solid_harm_func_by_ang(ang)(x, y, z).shape )
        x = np.array([1])
        y = np.array([1])
        z = np.array([1])
        print(find_solid_harm_func_by_ang(ang)(x, y, z) )
        x = np.random.randn(1000, 1000)
        y = np.random.randn(1000, 1000)
        z = np.random.randn(1000, 1000)
        print(find_solid_harm_func_by_ang(ang)(x, y, z).shape )       

        device = 'cuda'
        x = torch.randn(10, device=device)
        y = torch.randn(10, device=device)
        z = torch.randn(10, device=device)
        print(find_solid_harm_func_by_ang(ang)(x, y, z, package='torch', device=device).shape )
        x = torch.tensor([1], device=device)
        y = torch.tensor([1], device=device)
        z = torch.tensor([1], device=device)
        print(find_solid_harm_func_by_ang(ang)(x, y, z, package='torch', device=device) )
        x = torch.randn((1000, 1000), device=device)
        y = torch.randn((1000, 1000), device=device)
        z = torch.randn((1000, 1000), device=device)
        print(find_solid_harm_func_by_ang(ang)(x, y, z, package='torch', device=device).shape )
        
if __name__ == '__main__':
    test()


#     case 'd'
#         dM = sparse( ...
#             [1, 1, 1, 2, 3, 4, 4, 5]', ...
#             [1, 4, 6, 3, 5, 1, 4, 2]', ...
#             [-1/2, -1/2, 1, 1, 1, sqrt(3)/2, -sqrt(3)/2, 1]', ...
#             5, 6);
        
#         solid_harm_at_this_position = dM * [x*x; sqrt(3)*x*y; sqrt(3)*x*z; y*y; sqrt(3)*y*z; z*z];
#         return
        
#     case 'f'
#         fM = sparse( ...
#             [1, 1, 1, ...
#             2, 2, 2, ...
#             3, 3, 3, ...
#             4, 4, ...
#             5, ...
#             6, 6, ...
#             7, 7]', ...
#             ...
#             [3, 8, 10, ...
#             1, 4, 6, ...
#             2, 7, 9, ...
#             3, 8, ...
#             5, ...
#             1, 4, ...
#             2, 7]', ...
#             ...
#             [-3*sqrt(5)/10, -3*sqrt(5)/10, 1, ...
#             -sqrt(6)/4, -sqrt(30)/20, sqrt(30)/5, ...
#             -sqrt(30)/20, -sqrt(6)/4, sqrt(30)/5, ...
#             sqrt(3)/2, -sqrt(3)/2, ...
#             1, ...
#             sqrt(10)/4, -3*sqrt(2)/4, ...
#             3*sqrt(2)/4, -sqrt(10)/4]', ...
#             ...
#             7, 10);
        
#         solid_harm_at_this_position = fM * ...
#             [x*x*x; sqrt(5)*x*x*y; sqrt(5)*x*x*z; sqrt(5)*x*y*y; sqrt(15)*x*y*z; sqrt(5)*x*z*z; ...
#             y*y*y; sqrt(5)*y*y*z; sqrt(5)*y*z*z; z*z*z ];
        
#         return
        
#     case 'g'
#         gM = sparse( ...
#             [1, 1, 1, 1, 1, 1, ...
#             2, 2, 2, ...
#             3, 3, 3, ...
#             4, 4, 4, 4, ...
#             5 ,5, 5, ...
#             6, 6, ...
#             7, 7, ...
#             8, 8, 8, ...
#             9, 9]', ...
#             ...
#             [1, 4, 6, 11, 13, 15, ...
#             3, 8, 10, ...
#             5, 12, 14, ...
#             1, 6, 11, 13, ...
#             2, 7, 9, ...
#             3, 8, ...
#             5, 12, ...
#             1, 4, 11, ...
#             2, 7]', ...
#             ...
#             [3/8, 3*sqrt(105)/140, -3*sqrt(105)/35, 3/8 -3*sqrt(105)/35, 1, ...
#             -3*sqrt(70)/28, -3*sqrt(14)/28, sqrt(70)/7, ...
#             -3*sqrt(14)/28, -3*sqrt(70)/28, sqrt(70)/7, ...
#             -sqrt(5)/4, 3*sqrt(21)/4, sqrt(5)/4, -3*sqrt(21)/4, ...
#             -sqrt(35)/14, -sqrt(35)/14, 3*sqrt(7)/7, ...
#             sqrt(10)/4, -3*sqrt(2)/4, ...
#             3*sqrt(2)/4', -sqrt(10)/4, ...
#             sqrt(35)/8, -3*sqrt(3)/4, sqrt(35)/8, ...
#             sqrt(5)/2, -sqrt(5)/2]', ...
#             ...
#             9, 15);
        
#         solid_harm_at_this_position = gM * ...
#             [x*x*x*x; sqrt(7)*x*x*x*y; sqrt(7)*x*x*x*z; sqrt(35/3)*x*x*y*y; sqrt(35)*x*x*y*z; sqrt(35/3)*x*x*z*z; ...
#             sqrt(7)*x*y*y*y; sqrt(35)*x*y*y*z; sqrt(35)*x*y*z*z; sqrt(7)*x*z*z*z; ...
#             y*y*y*y; sqrt(7)*y*y*y*z; sqrt(35/3)*y*y*z*z; sqrt(7)*y*z*z*z; z*z*z*z ];
        
#         return
        
#     case 'h'
#         hM = sparse( ...
#             [1, 1, 1, 1, 1, 1, ...
#             2, 2, 2, 2, 2, 2, ...
#             3, 3, 3, 3, 3, 3, ...
#             4, 4, 4, 4, ...
#             5 ,5, 5, ...
#             6, 6, 6, 6, 6, ...
#             7, 7, 7, 7, 7, ...
#             8, 8, 8, ...
#             9, 9, ...
#             10, 10, 10, ...
#             11, 11, 11 ]', ...
#             ...
#             [3, 8, 10, 17, 19, 21, ...
#             1, 4, 6, 11, 13, 15, ...
#             2, 7, 9, 16, 18, 20, ...
#             3, 10, 17, 19, ...
#             5, 12, 14, ...
#             1, 4, 6, 11, 13, ...
#             2, 7, 9, 16, 18, ...
#             3, 8, 17, ...
#             5, 12, ...
#             1, 4, 11, ...
#             2, 7, 16 ]', ...
#             ...
#             [5/8, sqrt(105)/28, -5/sqrt(21), 5/8, -5/sqrt(21), 1, ...
#             sqrt(15)/8, sqrt(35)/28, -3*sqrt(35)/14, sqrt(15)/24, -3*sqrt(7)/14, sqrt(15)/3, ...
#             sqrt(15)/24, sqrt(35)/28, -3*sqrt(7)/14, sqrt(15)/8, -3*sqrt(35)/14, sqrt(15)/3, ...
#             -sqrt(105)/12, sqrt(5)/2, sqrt(105)/12, -sqrt(5)/2, ...
#             -sqrt(15)/6, -sqrt(15)/6, sqrt(15)/3, ...
#             -sqrt(70)/16, sqrt(30)/24, sqrt(30)/6, sqrt(70)/16, -sqrt(6)/2, ...
#             -sqrt(70)/16, -sqrt(30)/24, sqrt(6)/2, sqrt(70)/16, -sqrt(30)/6, ...
#             sqrt(35)/8, -3*sqrt(3)/4, sqrt(35)/8, ...
#             sqrt(5)/2, -sqrt(5)/2, ...
#             3*sqrt(14)/16, -5*sqrt(6)/8, 5*sqrt(14)/16, ...
#             5*sqrt(14)/16, -5*sqrt(6)/8, 3*sqrt(14)/16 ]', ...
#             ...
#             11, 21);
        
#         solid_harm_at_this_position = hM * ...
#             [x*x*x*x*x; 3*x*x*x*x*y; 3*x*x*x*x*z; 3*sqrt(7/3)*x*x*x*y*y; 3*sqrt(7)*x*x*x*y*z; 3*sqrt(7/3)*x*x*x*z*z; ...
#             3*sqrt(7/3)*x*x*y*y*y; 3*sqrt(35/3)*x*x*y*y*z; 3*sqrt(35/3)*x*x*y*z*z; 3*sqrt(7/3)*x*x*z*z*z; ...
#             3*x*y*y*y*y; 3*sqrt(7)*x*y*y*y*z; 3*sqrt(35/3)*x*y*y*z*z; 3*sqrt(7)*x*y*z*z*z; 3*x*z*z*z*z; ...
#             y*y*y*y*y; 3*y*y*y*y*z; 3*sqrt(7/3)*y*y*y*z*z; 3*sqrt(7/3)*y*y*z*z*z; 3*y*z*z*z*z; z*z*z*z*z ];
    
#     otherwise
#         disp('solid_harm.m: unrecognized solid harmonic');

