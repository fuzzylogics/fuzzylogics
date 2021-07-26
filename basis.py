# basis
import enum
import os
import re
import torch
from torch import autograd
import numpy as np
from numpy import sqrt, prod
from solid_harms import find_solid_harm_func_by_ang
import time
import opt_einsum

# class Basis():
#     def __init__(self, centers=[0], ):
#         self.params = 

CPU_MEM = 4.0e10
GPU_MEM = 2.0e10


class BasisSet():

    def __init__(self, params_list=None, filepath=None, atom=None, centers=None, rotations=None):
        self.params_list = []
        if params_list is not None:
            self.params_list = params_list   
        elif filepath is not None:
            self.params_list = self.read_params(filepath, atom)
            self.normalize()
            self.define_center_and_rotation(centers, rotations)
        

    def read_params(self, fp, atom):
        params_list = []
        assert fp.rsplit('.', 1)[1] == 'gbs'
        with open(fp) as f:
            line = f.readline()
            while line:
                words = line.split()
                if len(words) > 0 and words[0] == atom and len(words) == 2:
                    words = f.readline().split()
                    while len(words)==3 and words[0] in ('S','P','D','F','G','H','s','p','d','f','g','h'):
                        ang, n_contraction = words[0].upper(), int(words[1])
                        params_list.append({'ang':ang,'n_contraction':n_contraction,'exponents':[],'front_factors':[]})    
                        for i in range(n_contraction):
                            exponent, front_factor = f.readline().split()
                            exponent, front_factor = float(re.sub('D|d','e',exponent)), float(re.sub('D|d','e',front_factor))
                            params_list[-1]['exponents'].append(exponent)
                            params_list[-1]['front_factors'].append(front_factor)
                        params_list[-1]['exponents'] = np.array(params_list[-1]['exponents'])
                        params_list[-1]['front_factors'] = np.array(params_list[-1]['front_factors'])                       
                        words = f.readline().split()
                line = f.readline()
        return params_list

    #def write_params(fn, params_list):

    def define_center_and_rotation(self, centers=None, rotations=None):
        if centers == None: centers = (0.0,0.0,0.0)
        if rotations == None: rotations = (0.0,0.0,0.0,0.0)
        if len(centers)==3 and type(centers[0])==float: centers = [centers]*len(self.params_list)
        if type(rotations) is float or (len(rotations) in (1,3,4)) and type(rotations[0])==float:
            rot_dict = {'around_z': None, 'z_to':None}
            if type(rotations) is float: rot_dict['around_z'] = rotations
            elif len(rotations) == 1: rot_dict['around_z'] = rotations[0]
            elif len(rotations) == 3: rot_dict['z_to'] = np.array(rotations)
            elif len(rotations) == 4:
                rot_dict['around_z'] = rotations[0]
                rot_dict['z_to'] = np.array(rotations[1:4])
            else:
                raise NotImplementedError('What rotation %s?'%(rotations))
            rot_dicts = [rot_dict]*len(self.params_list)
        elif type(rotations) is dict:
            rot_dicts = [rotations]*len(self.params_list)
        else: raise NotImplementedError('rotations ill-defined')       
        assert len(centers) == len(self.params_list) and len(rot_dicts) == len(self.params_list)
        for i, gaussian_dict in enumerate(self.params_list):
            gaussian_dict['center'] = np.array(centers[i])
            gaussian_dict['rot_dict'] = rot_dicts[i]
            # gaussian_dict['rotation_vector'] = np.array(rotations[i])
            
    def normalize(self):        
        def normalize_factor(a, l, a2=None):
            ang_num = {'S':0,'P':1,'D':2,'F':3,'G':4,'H':5,
                       's':0,'p':1,'d':2,'f':3,'g':4,'h':5}
            if type(l) is str:  l = ang_num[l]
            if a2 == None:  a2 = a
            return sqrt( ((a+a2)/np.pi)**(3/2) * (2*(a+a2))**l / prod(range(2*l-1,0,-2)) )
        for gaussian_dict in self.params_list:
            ang = gaussian_dict['ang']
            n_contraction = gaussian_dict['n_contraction']
            if n_contraction == 1:
                gaussian_dict['front_factors'][0] *= normalize_factor(gaussian_dict['exponents'][0], ang)
            else:
                #contaction_weights = gaussian_dict['front_factors']
                gaussian_dict['front_factors'] *= normalize_factor(gaussian_dict['exponents'], ang)
                N = 0.0
                for i in range(n_contraction):
                    for j in range(n_contraction):
                        integral = gaussian_dict['front_factors'][i] * gaussian_dict['front_factors'][j]
                        integral *= 1.0 / normalize_factor(gaussian_dict['exponents'][i], ang, gaussian_dict['exponents'][j])**2
                        N += integral
                N = 1.0 / sqrt(N)
                gaussian_dict['front_factors'] *= N

    @property
    def n_orbs(self):
        n_orbs = 0
        for gaussian_dict in self.params_list:
            n_orbs += self.ang_n_orbs(gaussian_dict['ang'])
        return n_orbs

    @property
    def n_g(self):
        return len(self.params_list)

    @classmethod
    def ang_num(cls, ang):
        ang_num_dict = {'S':0,'P':1,'D':2,'F':3,'G':4,'H':5,
                        's':0,'p':1,'d':2,'f':3,'g':4,'h':5}
        return ang_num_dict[ang]
    @classmethod
    def ang_n_orbs(cls, ang):
        return cls.ang_num(ang)*2 + 1

    @staticmethod
    def multiply_gaussians(g1, g2):
        a1, f1 = g1['exponents'], g1['front_factors']
        if 'centers' in g1 and g1['centers'] is not None: r1 = g1['centers']
        else: r1 = np.repeat(np.expand_dims(g1['center'], 0), g1['n_contraction'], axis=0)
        a2, f2 = g2['exponents'], g2['front_factors']
        if 'centers' in g2 and g2['centers'] is not None: r2 = g2['centers']
        else: r2 = np.repeat(np.expand_dims(g2['center'], 0), g2['n_contraction'], axis=0)
        a, r, f = BasisSet.gXg_w_torch(a1, f1, r1, a2, f2, r2)
        return {'exponents':a, 'front_factors':f, 'centers':r}     
    @staticmethod
    def gXg_w_torch(a1, f1, r1, a2, f2, r2):
        # all args on the same device
        a = a1.view(a1.shape+(1,)*len(a2.shape)) + a2.view((1,)*len(a1.shape)+a2.shape)
        assert a1.shape == r1.shape[0:-1] and r1.shape[-1] == 3
        assert a2.shape == r2.shape[0:-1] and r2.shape[-1] == 3
        a1r1 = (a1.unsqueeze(-1)*r1).view(a1.shape + (1,)*len(a2.shape) + (3,))
        a2r2 = (a2.unsqueeze(-1)*r2).view((1,)*len(a1.shape) + a2.shape + (3,))
        r = (a1r1 + a2r2) / a.unsqueeze(-1)
        r1_expanded = r1.view(a1.shape + (1,)*len(a2.shape) + (3,))
        r2_expanded = r2.view((1,)*len(a1.shape) + a2.shape + (3,))
        a1a2 = a1.view(a1.shape+(1,)*len(a2.shape)) * a2.view((1,)*len(a1.shape)+a2.shape)
        f1_expanded = f1.view(f1.shape+(1,)*len(f2.shape))
        f2_expanded = f2.view((1,)*len(f1.shape)+f2.shape)
        f = f1_expanded * f2_expanded * torch.exp(-(a1a2/a) * torch.sum((r1_expanded-r2_expanded)**2, -1) )
        return a, r, f

    def concat_gauss(self, package='numpy', device='cpu'):
        self.concated_gauss = {}
        # basis is rather small, efficiency is not priority
        a_s, f_s, r_s, n_angs = [], [], [], []
        ang_dict = {'S':1,'P':3,'D':5,'F':7,'G':9,'H':11,'s':1,'p':3,'d':5,'f':7,'g':9,'h':11}
        ctracs = [0] #contraction starts with 0 and has n+1 elements
        n_gauss, n_prime = 0, 0
        for g in self.params_list:
            n_prime += g['n_contraction']
            ctracs += [n_prime]
            n_gauss += 1
            for i_ctrac in range(g['n_contraction']):
                a_s.append(g['exponents'][i_ctrac])
                f_s.append(g['front_factors'][i_ctrac])
            if 'centers' in g:
                assert tuple(g['centers'].shape) == (g['n_contraction'], 3)
                for i_ctrac in range(g['n_contraction']):            
                    r_s.append(tuple(g['centers'][i_ctrac]))
            elif 'center' in g:
                assert tuple(g['center'].shape) == (3,)
                r_s += [g['center']] * g['n_contraction']
            # n_angs += [[ang_dict[g['ang']]]] * g['n_contraction'] # last layer for nesting
            n_angs += [ang_dict[g['ang']]] * g['n_contraction']
        self.concated_gauss['n_gauss'] = n_gauss
        self.concated_gauss['n_prime'] = n_prime
        self.concated_gauss['contractions'] = (tuple(ctracs), ) # tuple of tuple
        if package == 'numpy':
            self.concated_gauss['exponents'] = np.array(a_s)
            self.concated_gauss['front_factors'] = np.array(f_s)
            self.concated_gauss['centers'] = np.array(r_s)
            self.concated_gauss['ori_centers'] = (np.array(r_s),)
            self.concated_gauss['n_angs'] = (np.array(n_angs, dtype=int), ) # tuple
            # self.concated_gauss['contractions'] = np.array(ctracs)
        elif package == 'torch':
            self.concated_gauss['exponents'] = torch.tensor(a_s, dtype=torch.float32, device=device)
            self.concated_gauss['front_factors'] = torch.tensor(f_s, dtype=torch.float32, device=device)
            self.concated_gauss['centers'] = torch.tensor(r_s, dtype=torch.float32, device=device)
            self.concated_gauss['ori_centers'] = (torch.tensor(r_s, dtype=torch.float32, device=device),)
            self.concated_gauss['n_angs'] = (torch.tensor(n_angs, dtype=torch.long, device=device), ) # tuple
            # self.concated_gauss['contractions'] = torch.tensor(ctracs, dtype=torch.long, device=device)
        else: raise NotImplementedError('numpy or torch ?')
        return self.concated_gauss
    
    @staticmethod
    def multiply_batch_gauss_torch(g1, g2):
        n_gauss = g1['n_gauss'] * g2['n_gauss']
        n_prime = g1['n_prime'] * g2['n_prime']
        ctracs = g1['contractions'] + g2['contractions'] # concat tuples
        n_angs = g1['n_angs'] + g2['n_angs'] # concat tuples
        ori_centers = g1['ori_centers'] + g2['ori_centers'] # concat tuples
        # n_angs_shape1 = g1['n_angs'].shape[:-1]+(1,)*(len(g2['n_angs'].shape)-1)+(g1['n_angs'].shape[-1],)
        # n_angs_shape2 = (1,)*(len(g1['n_angs'].shape)-1)+g2['n_angs'].shape
        # n_angs = torch.cat( ( g1['n_angs'].view(n_angs_shape1).repeat(n_angs_shape2[:-1]+(1,)),
        #                       g2['n_angs'].view(n_angs_shape2).repeat(n_angs_shape1[:-1]+(1,)) ), -1)
        a1, f1, r1 = g1['exponents'], g1['front_factors'], g1['centers']
        a2, f2, r2 = g2['exponents'], g2['front_factors'], g2['centers'] 
        a, r, f = BasisSet.gXg_w_torch(a1, f1, r1, a2, f2, r2) # all args should on the same device
        return {'exponents':a, 'front_factors':f, 
                'centers':r, 'ori_centers': ori_centers,
                'n_angs':n_angs,
                'n_gauss':n_gauss, 'n_prime':n_prime, 'contractions':ctracs}
    @staticmethod
    def multiply_batch_gauss_moving_center_torch(g1, g2, g2c, collapes_a=False): # g2c is tensor of centers
        n_gauss = g1['n_gauss'] * g2['n_gauss'] * torch.tensor(g2c.shape[:-1]).prod()
        n_prime = g1['n_prime'] * g2['n_prime'] * torch.tensor(g2c.shape[:-1]).prod()
        ctracs = g1['contractions'] + g2['contractions']
        ori_centers = g1['ori_centers'] + g2['ori_centers'] + (g2c,)
        moving_centers_on_dims = {2:[i+3 for i in range(len(g2c.shape))]}
        n_angs = g1['n_angs'] + g2['n_angs'] 
        a1, f1, r1 = g1['exponents'], g1['front_factors'], g1['centers']
        a2, f2, r2 = g2['exponents'], g2['front_factors'], g2['centers']
        g2c_expended = g2c.view((1,)*(len(r2.shape)-1)+g2c.shape)
        a2 = a2.view(a2.shape+(1,)*len(g2c.shape[:-1]))
        f2 = f2.view(f2.shape+(1,)*len(g2c.shape[:-1]))
        r2 = r2.view(r2.shape[:-1]+(1,)*len(g2c.shape[:-1])+(3,))
        a2 = a2 + torch.zeros_like(g2c_expended[...,0]) # '+=' doesn't broadcast
        f2 = f2 + torch.zeros_like(g2c_expended[...,0])
        r2 = r2 + g2c_expended
        # print(a1.shape, f1.shape, r1.shape, a2.shape, f2.shape, r2.shape)
        a, r, f = BasisSet.gXg_w_torch(a1, f1, r1, a2, f2, r2)
        if collapes_a : a = a.mean([*range(-len(g2c.shape)+1,0)], keep_dim=True)
        #     for i in range(len(g2c.shape)-1):
        #         a = a[...,0]
        return {'exponents':a, 'front_factors':f, 
                'centers':r, 'ori_centers': ori_centers, 'moving_centers_on_dims': moving_centers_on_dims,
                'n_angs':n_angs,
                'n_gauss':n_gauss, 'n_prime':n_prime, 'contractions':ctracs}


    # @staticmethod
    # wrong! normalization constant normalize basis squired
    # def integrate_poly_gauss_wof_torch(pw, a):
    #     # integrate polynomial times gaussian without front factors
    #     sz = tuple(a.shape)
    #     assert tuple(pw.shape) == sz+(3,)
    #     # double factorials
    #     max_order = 5 # faster maybe?
    #     assert pw.max() <= max_order
    #     pw_fact = torch.ones_like(pw)
    #     tmp_fact = 2*pw - 1
    #     for i in range(max_order):
    #         tmp_fact[tmp_fact<1] = 1
    #         pw_fact *= tmp_fact
    #         tmp_fact -= 2
    #     numerator = torch.prod(pw_fact, dim=-1)       
    #     denominator = torch.sqrt((2*a/torch.pi)**3) * torch.pow(4*a, torch.sum(pw, dim=-1))
    #     return torch.sqrt(numerator/denominator)
    @staticmethod
    def integrate_stripped_polyXgauss_torch(pw, a):
        # x**pw[0] * y**pw[1] * z**pw[2] * exp(-a * r**2)
        assert type(pw) == torch.Tensor
        assert type(a) == torch.Tensor
        assert pw.device == a.device
        sz = tuple(a.shape)
        assert tuple(pw.shape) == sz+(3,)
        pw = pw.int()
        res = torch.empty_like(pw, dtype=torch.float32)
        res[pw%2==1] = 0.0
        res[pw%2==0] = torch.exp( torch.lgamma((pw[pw%2==0]+1)/2) )
        res = res.prod(-1)
        res *= a**(-(pw.sum(-1)+3)/2)
        return res

    # @staticmethod
    # def integrate_polyXgauss_torch_vanilla(g, device='cpu'):
    #     # g is concated gauss
    #     # ro = g['ori_centers']
    #     # a, f, r = g['exponents'], g['front_factors'], g['centers']
    #     # for dim in range(len(g['n_angs'])):
    #     #     a = a.repeat_interleave(g['n_angs'][dim], dim=dim)
    #     from solid_harms_sm import solid_harm_poly_LS
    #     highest_ang = max([n_angs.max().item() for n_angs in g['n_angs']])
    #     sh_poly = solid_harm_poly_LS(highest_ang, trans_dict={'translation':True}, 
    #                                  folder='shs', to_str=True, to_torch=True, 
    #                                  load_sh=True, save_sh=True) # dict of dict after flatten
    #     # print([ang for ang in sh_poly.keys()])
    #     # print(sh_poly)
    #     n_poly_lookup = {ang:len(sh_poly[ang]['poly_indces']) for ang in sh_poly.keys()}
    #     n_poly = tuple([ torch.tensor([n_poly_lookup[ang.item()] for ang in angs])
    #                      for angs in g['n_angs'] ])
    #     poly_idx = tuple([ torch.tensor([0,]+[n_poly_lookup[ang.item()] for ang in angs]).cumsum(0) 
    #                        for angs in g['n_angs'] ])
        
    #     f_expanded2poly = torch.ones( tuple([sum([n_poly_lookup[ang.item()] for ang in angs]) for angs in g['n_angs']]),  
    #                                   dtype=torch.float32, device=device ) # contraction not done yet
    #     pw_expanded2poly = torch.zeros( tuple([sum([n_poly_lookup[ang.item()] for ang in angs])
    #                                            for angs in g['n_angs']]) + (3,),  
    #                                     dtype=torch.int, device=device )    
    #     # print(g['contractions'])
    #     for dim, angs in enumerate(g['n_angs']):
    #         assert all(angs.sort(0)[0] == angs)
    #         ro_expand_dim = [1,]*(len(g['centers'].shape)-1) + [3,]
    #         ro_expand_dim[dim] = g['ori_centers'][dim].shape[0]
    #         r_ref = g['ori_centers'][dim].view(ro_expand_dim) - g['centers']
    #         # front factors from r
    #         for ang in [1, 3, 5, 7, 9, 11]:
    #             print(ang)
    #             if ang in angs:                   
    #                 bg, ed = (angs==ang).nonzero(as_tuple=True)[0][0].item(), (angs==ang).nonzero(as_tuple=True)[0][-1].item() # bg, ed are indces of gaussians               
                    
    #                 x_ref, y_ref, z_ref = [t.squeeze(-1) for t in r_ref.narrow(dim, bg, ed+1-bg).split(1, dim=-1)]
    #                 for i_poly in range(n_poly_lookup[ang]):
    #                     tmp_idx = range(poly_idx[dim][bg]+i_poly, poly_idx[dim][ed+1], n_poly_lookup[ang])
                        
    #                     print(sh_poly[ang]['coeffs'][i_poly])
    #                     eval_ith_poly_f = eval(sh_poly[ang]['coeffs'][i_poly])
    #                     print(eval_ith_poly_f)
    #                     ith_poly_pw = torch.tensor(sh_poly[ang]['orders'][i_poly], dtype=torch.int, device=device)
    #                     for d in range(len(n_poly)): ith_poly_pw = ith_poly_pw.unsqueeze(0)
    #                     # print(ith_poly_pw)
    #                     if type(eval_ith_poly_f) in (float, int) or eval_ith_poly_f.shape == torch.Size([]):
    #                         eval_ith_poly_f = eval_ith_poly_f * torch.ones_like(x_ref)
    #                     print(eval_ith_poly_f.shape)
    #                     for d in range(len(n_poly)):
    #                         if d != dim: eval_ith_poly_f = eval_ith_poly_f.repeat_interleave(n_poly[d], dim=d)
    #                     for ig, idx in enumerate(tmp_idx):    
    #                         f_expanded2poly.narrow(dim, idx, 1)[:] *= eval_ith_poly_f.narrow(dim, ig, 1)
    #                         pw_expanded2poly.narrow(dim, idx, 1)[:] += ith_poly_pw
    #                         # print(pw_expanded2poly.narrow(dim, idx, 1).shape)
    #                         # print(ith_poly_pw.shape)
        
    #     a, f_g = g['exponents'], g['front_factors'],
    #     for dim in range(len(n_poly)):
    #         a = a.repeat_interleave(n_poly[dim], dim=dim)        
    #         f_g = f_g.repeat_interleave(n_poly[dim], dim=dim)

    #     res_expended2poly = f_g * f_expanded2poly * \
    #                         BasisSet.integrate_stripped_polyXgauss_torch(pw_expanded2poly, a)
    #     for ang in sh_poly:
    #         pi = torch.tensor(sh_poly[ang]['poly_indces'])
    #         _, inv_poly_idx = pi.unique(sorted=True, return_inverse=True)
    #         assert all(inv_poly_idx == pi)
    #     res_uncontracted = res_expended2poly
    #     for dim, angs in enumerate(g['n_angs']):
    #         tmp_idx, c = torch.tensor([], dtype=torch.int), 0
    #         for ang in angs.tolist():
    #             pi = torch.tensor(sh_poly[ang]['poly_indces'])
    #             tmp_idx = torch.cat([tmp_idx, pi + c])
    #             c += sh_poly[ang]['poly_indces'][-1] + 1
    #         tmp = torch.zeros_like(res_uncontracted.narrow(dim, 0, tmp_idx[-1]+1))
    #         tmp.index_add_(dim, tmp_idx ,res_uncontracted)
    #         res_uncontracted = tmp

    #     res = res_uncontracted
    #     for dim, ctrac in enumerate(g['contractions']):
    #         tmp_idx4ctrac, c = [], 0
    #         for i in range(len(ctrac)-1):
    #             tmp_idx4ctrac += [*range(c, c+g['n_angs'][dim][ctrac[i]])]*(ctrac[i+1]-ctrac[i])
    #             c += g['n_angs'][dim][ctrac[i]]
    #         tmp_idx4ctrac = torch.tensor(tmp_idx4ctrac)
    #         tmp = torch.zeros_like(res.narrow(dim, 0, tmp_idx4ctrac[-1]+1))
    #         tmp.index_add_(dim, tmp_idx4ctrac ,res)
    #         res = tmp

    #     return res


    def gauss_on_grid(self, coords, contract=True, device='cpu'):
        g = self.concat_gauss(package='torch', device=device)
        grid_dims = coords.shape[:-1]
        assert coords.shape[-1] == 3
        a, f, rc = g['exponents'], g['front_factors'], g['centers']
        g_dims = a.shape
        assert f.shape == g_dims
        assert rc.shape == g_dims + (3,)
        a, f = a.view(g_dims+(1,)*len(grid_dims)), f.view(g_dims+(1,)*len(grid_dims))
        rc = rc.view(g_dims+(1,)*len(grid_dims)+(3,))
        r = coords.view((1,)*len(g_dims)+grid_dims+(3,))
        r = r - rc
        r2 = (r**2).sum(-1)
        res = f * torch.exp(-a * r2)
        if contract:
            for dim, ctrac in enumerate(g['contractions']):
                tmp_idx4ctrac, c = [], 0
                for i in range(len(ctrac)-1):
                    tmp_idx4ctrac += [c,]*(ctrac[i+1]-ctrac[i])
                    c += 1
                tmp_idx4ctrac = torch.tensor(tmp_idx4ctrac)
                tmp = torch.zeros_like(res.narrow(dim, 0, tmp_idx4ctrac[-1]+1))
                tmp.index_add_(dim, tmp_idx4ctrac ,res)
                res = tmp
        return res                

    @staticmethod
    def integrate_polyXgauss_torch(g, device='cpu'):
        # g is concated gauss

        if 'moving_center_on_dims' in g: mc_dims = g['moving_center_on_dims']
        else: mc_dims = {}
        n_mc_dims = sum(len(v) for v in list(mc_dims.values()))

        from solid_harms_sm import solid_harm_poly_LS
        highest_ang = max([n_angs.max().item() for n_angs in g['n_angs']])
        sh_poly = solid_harm_poly_LS(highest_ang, trans_dict={'translation':True}, 
                                     folder='shs', to_str=True, to_torch=True, 
                                     load_sh=True, save_sh=True) # dict of dict after flatten
        # print([ang for ang in sh_poly.keys()])
        # print(sh_poly)
        n_poly_lookup = {ang:len(sh_poly[ang]['poly_indces']) for ang in sh_poly.keys()}
        n_poly = tuple([ torch.tensor([n_poly_lookup[ang.item()] for ang in angs])
                         for angs in g['n_angs'] ])
        poly_idx = tuple([ torch.tensor([0,]+[n_poly_lookup[ang.item()] for ang in angs]).cumsum(0) 
                           for angs in g['n_angs'] ])
        
        f_expanded2poly = torch.ones( tuple([sum([n_poly_lookup[ang.item()] for ang in angs]) 
                                             for angs in g['n_angs']]) +
                                             (1,)*n_mc_dims,
                                      dtype=torch.float32, device=device ) # contraction not done yet
        pw_expanded2poly = torch.zeros( tuple([sum([n_poly_lookup[ang.item()] for ang in angs])
                                               for angs in g['n_angs']]) +
                                               (1,)*n_mc_dims + (3,),  
                                        dtype=torch.int, device=device )    
        # print(g['contractions'])
        for dim, angs in enumerate(g['n_angs']):
            assert all(angs.sort(0)[0] == angs)
            # ro
            ro_expand_dim = [1,]*(len(g['centers'].shape)-1) + [3,]
            ro_expand_dim[dim] = g['ori_centers'][dim].shape[0]
            r_ref = g['ori_centers'][dim].view(ro_expand_dim) - g['centers']
            # mc
            mc_expand_dim = [1,]*(len(g['centers'].shape)-1) + [3,]
            if dim in mc_dims:
                for d in mc_dims[dim]: mc_expand_dim[d] = g['centers'].shape[d]
                r_ref = r_ref + g['ori_centers'][ len(g['n_angs'])+list(mc_dims.keys()).index(dim) ].view(mc_expand_dim)

            # front factors from r
            for ang in [1, 3, 5, 7, 9, 11]:
                if ang in angs:                   
                    bg, ed = (angs==ang).nonzero(as_tuple=True)[0][0].item(), (angs==ang).nonzero(as_tuple=True)[0][-1].item() # bg, ed are indces of gaussians               
                    
                    x_ref, y_ref, z_ref = [t.squeeze(-1) for t in r_ref.narrow(dim, bg, ed+1-bg).split(1, dim=-1)]
                    for i_poly in range(n_poly_lookup[ang]):
                        tmp_idx = range(poly_idx[dim][bg]+i_poly, poly_idx[dim][ed+1], n_poly_lookup[ang])
                        
                        print(sh_poly[ang]['coeffs'][i_poly])
                        eval_ith_poly_f = eval(sh_poly[ang]['coeffs'][i_poly])
                        print(eval_ith_poly_f)
                        ith_poly_pw = torch.tensor(sh_poly[ang]['orders'][i_poly], dtype=torch.int, device=device)
                        for d in range(len(n_poly)): ith_poly_pw = ith_poly_pw.unsqueeze(0)
                        # print(ith_poly_pw)
                        if type(eval_ith_poly_f) in (float, int) or eval_ith_poly_f.shape == torch.Size([]):
                            eval_ith_poly_f = eval_ith_poly_f * torch.ones_like(x_ref)
                        print(eval_ith_poly_f.shape)
                        for d in range(len(n_poly)):
                            if d != dim: eval_ith_poly_f = eval_ith_poly_f.repeat_interleave(n_poly[d], dim=d)
                        for ig, idx in enumerate(tmp_idx):    
                            f_expanded2poly.narrow(dim, idx, 1)[:] *= eval_ith_poly_f.narrow(dim, ig, 1)
                            pw_expanded2poly.narrow(dim, idx, 1)[:] += ith_poly_pw
                            # print(pw_expanded2poly.narrow(dim, idx, 1).shape)
                            # print(ith_poly_pw.shape)
        
        a, f_g = g['exponents'], g['front_factors'],
        for dim in range(len(n_poly)):
            a = a.repeat_interleave(n_poly[dim], dim=dim)        
            f_g = f_g.repeat_interleave(n_poly[dim], dim=dim)

        res_expended2poly = f_g * f_expanded2poly * \
                            BasisSet.integrate_stripped_polyXgauss_torch(pw_expanded2poly, a)
        for ang in sh_poly:
            pi = torch.tensor(sh_poly[ang]['poly_indces'])
            _, inv_poly_idx = pi.unique(sorted=True, return_inverse=True)
            assert all(inv_poly_idx == pi)
        res_uncontracted = res_expended2poly
        for dim, angs in enumerate(g['n_angs']):
            tmp_idx, c = torch.tensor([], dtype=torch.int), 0
            for ang in angs.tolist():
                pi = torch.tensor(sh_poly[ang]['poly_indces'])
                tmp_idx = torch.cat([tmp_idx, pi + c])
                c += sh_poly[ang]['poly_indces'][-1] + 1
            tmp = torch.zeros_like(res_uncontracted.narrow(dim, 0, tmp_idx[-1]+1))
            tmp.index_add_(dim, tmp_idx ,res_uncontracted)
            res_uncontracted = tmp

        res = res_uncontracted
        for dim, ctrac in enumerate(g['contractions']):
            tmp_idx4ctrac, c = [], 0
            for i in range(len(ctrac)-1):
                tmp_idx4ctrac += [*range(c, c+g['n_angs'][dim][ctrac[i]])]*(ctrac[i+1]-ctrac[i])
                c += g['n_angs'][dim][ctrac[i]]
            tmp_idx4ctrac = torch.tensor(tmp_idx4ctrac)
            tmp = torch.zeros_like(res.narrow(dim, 0, tmp_idx4ctrac[-1]+1))
            tmp.index_add_(dim, tmp_idx4ctrac ,res)
            res = tmp

        return res



    # def itg_ovlp_torch(self, other, device='cpu'):
    #     g1 = self.concat_gauss(package='torch', device=device)
    #     g2 = other.concat_gauss(package='torch', device=device)
    #     # angs1, angs2 = g1['n_angs'], g2['n_angs']
    #     g1Xg2 = self.multiply_batch_gauss_torch(g1, g2)
    #     assert len(g1Xg2['n_angs']) == 2   
    #     a, f, r = g1Xg2['exponents'], g1Xg2['front_factors'], g1Xg2['centers']
    #     for dim in range(len(g1Xg2['n_angs'])):
    #         a = a.repeat_interleave(g1Xg2['n_angs'][dim], dim=dim)
    #     from solid_harms_sm import solid_harm_poly_LS
    #     highest_ang = max([n_angs.max().item() for n_angs in g1Xg2['n_angs']])
    #     sh_poly = solid_harm_poly_LS(highest_ang, trans_dict={'translation':True}, 
    #                                      folder='shs', to_str=True, to_torch=True, 
    #                                      load_sh=True, save_sh=True) # dict of dict after flatten      
        
    #     res = torch.ones( [sum(angs) for angs in g1Xg2['n_angs']],  
    #                        dtype=torch.float32, device=device ) # contraction not done yet
        
    #     for dim, angs in enumerate(g1Xg2['n_angs']):
    #         assert all(angs.sort(0)[0] == angs)
    #         for 

        


        # for i_g, i_orbs in enumerate((0,)+tuple(angs1.cumsum(0)[:-1].tolist())):
        #     for j_g, j_orbs in enumerate((0,)+tuple(angs2.cumsum(0)[:-1].tolist())):
        #         print(res[i_orbs:i_orbs+angs1[i_g], j_orbs:j_orbs+angs2[j_g]].shape)


    


    # def integrate_shXgauss_torch(sh_poly, gauss):
    #     ...


    # def calculate_on_grid(self, grid_coords, package='numpy', device='cpu'):
    #     # sh_package = 'scipy' if package=='numpy' else package
    #     start_time_coords = time.time()
    #     if not isinstance(grid_coords, np.ndarray): grid_coords = np.array(grid_coords)
        
    #     if len(grid_coords.shape) == 1 and len(grid_coords) == 3:
    #         grid_coords = np.expand_dims(grid_coords, axis=0)
    #     x0, y0, z0 = np.split(grid_coords, 3, axis=-1)
    #     x0, y0, z0 = np.squeeze(x0,-1), np.squeeze(y0,-1), np.squeeze(z0,-1)
    #     coords_shape = tuple(x0.shape)
    #     n_orbs = self.n_orbs

    #     if package == 'numpy':
    #         on_grid_orbs = np.empty(coords_shape+(n_orbs,))
    #     elif package == 'torch':
    #         x0 = torch.from_numpy(x0).float().to(device)
    #         y0 = torch.from_numpy(y0).float().to(device)
    #         z0 = torch.from_numpy(z0).float().to(device)
    #         on_grid_orbs = torch.empty(coords_shape+(n_orbs,), dtype=torch.float32, device=device)
    #     r2_0 = x0**2 + y0**2 + z0**2
    #     rotate_dims = [-1,]+[i for i in range(len(coords_shape))]   # rotation used in matrix dims       
    #     if package == 'numpy':
    #         exp, sin, cos = np.exp, np.sin, np.cos
    #     elif package == 'torch':
    #         exp, sin, cos = torch.exp, torch.sin, torch.cos
    #     end_time_coords = time.time()
    #     orb_idx = 0
    #     for gaussian_dict in self.params_list:
    #         x, y, z, r2 = x0, y0, z0, r2_0
    #         alphas, Ns = gaussian_dict['exponents'], gaussian_dict['front_factors']
    #         sh_func = find_solid_harm_func_by_ang(gaussian_dict['ang'])
    #         if gaussian_dict['center'] != (0.0, 0.0, 0.0):
    #             x, y, z = x0-gaussian_dict['center'][0], y0-gaussian_dict['center'][1], z0-gaussian_dict['center'][2]
    #             r2 = x**2 + y**2 + z**2
    #         if gaussian_dict['rotation_vector'] != (0.0, 0.0, 0.0):
    #             rot_mat = Rotation.from_rotvec(gaussian_dict['rotation_vector']).as_matrix()
    #             if package=='numpy':
    #                 x, y, z = np.split(np.matmul(np.stack((x,y,z), axis=-1), rot_mat.T), 3, axis=-1)
    #                 x, y, z = np.squeeze(x,-1), np.squeeze(y,-1), np.squeeze(z,-1)
    #             elif package=='torch': 
    #                 rot_mat = torch.from_numpy(rot_mat).float().to(device)
    #                 x, y, z = torch.split(torch.matmul(torch.stack((x,y,z), axis=-1), rot_mat.T), 1, dim=-1)
    #                 x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)
    #         if package == 'numpy':
    #             gaussian = np.sum(Ns * exp(-np.matmul(np.expand_dims(r2,-1), np.expand_dims(alphas,-2))), -1)
    #             # on_grid_orbs_each_gaussian = sh_func(x, y, z, package='scipy').transpose(rotate_dims) * gaussian
    #             on_grid_orbs_each_gaussian = sh_func(x, y, z, package='scipy') * np.expand_dims(gaussian, -1)        
    #         elif package == 'torch':
    #             alphas = torch.from_numpy(alphas).float().to(device)
    #             Ns = torch.from_numpy(Ns).float().to(device)      
    #             # gaussian = torch.sum(Ns * torch.exp(-torch.outer(r2, alphas)), 1)
    #             gaussian = torch.sum(Ns * exp(-torch.matmul(r2.unsqueeze(-1), alphas.unsqueeze(-2))), -1)
    #             # on_grid_orbs_each_gaussian = sh_func(x, y, z, package='torch', device=device).permute(rotate_dims) * gaussian
    #             on_grid_orbs_each_gaussian = sh_func(x, y, z, package='torch', device=device) * gaussian.unsqueeze(-1)
    #         on_grid_orbs[... ,orb_idx:orb_idx+self.ang_n_orbs(gaussian_dict['ang'])] = on_grid_orbs_each_gaussian
    #         orb_idx += self.ang_n_orbs(gaussian_dict['ang'])
    #     end_time_orbs = time.time()
    #     print("Time for handling coordinate = ", end_time_coords-start_time_coords)
    #     print("Time for calculating on grid orbitals = ", end_time_orbs-end_time_coords)
    #     return on_grid_orbs


    # def on_grid_w_torch(self, grid_coords, deriv=0, device='cuda'):
    #     time0 = time.time()
    #     if isinstance(grid_coords, np.ndarray):
    #         grid_coords = torch.from_numpy(grid_coords).float().to(device)
    #     if deriv != 0: grid_coords.requires_grad = True
    #     x, y, z = torch.split(grid_coords, 1, dim=-1)
    #     x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)
    #     orbs = torch.empty(x.shape+(self.n_orbs,), dtype=torch.float32, device=device)
    #     # fill in result gaussian by gaussian
    #     time1 = time.time()
    #     orb_i = 0
    #     for g in self.params_list:
    #         xc, yc, zc, = g['center']
    #         a = torch.from_numpy(g['exponents']).float().to(device)
    #         N = torch.from_numpy(g['front_factors']).float().to(device)
    #         sh = find_solid_harm_func_by_ang(g['ang'])
    #         x2c, y2c, z2c = x-xc, y-yc, z-zc
    #         r2 = x2c**2 + y2c**2 + z2c**2
    #         gauss = torch.sum(N * torch.exp(-torch.matmul(r2.unsqueeze(-1), a.unsqueeze(0))), -1)
    #         shXgauss = sh(x2c, y2c, z2c, package='torch',device=device) * gauss.unsqueeze(-1)
    #         orbs.narrow(-1,orb_i,self.ang_n_orbs(g['ang']))[:] = shXgauss
    #         orb_i += self.ang_n_orbs(g['ang'])
    #     time2 = time.time()
    #     if deriv == 1:
    #         sum_dim = tuple(i for i in range(len(orbs.shape)-1))
    #         orbs_1 = torch.empty(grid_coords.shape+(self.n_orbs,), dtype=torch.float32, device=device)
    #         for i in range(self.n_orbs):
    #             v = torch.zeros(self.n_orbs, device=device)
    #             v[i] = 1.0
    #             # print('orbs_sum',orbs.sum(dim=sum_dim))
    #             orbs_1.narrow(-1,i,1)[:] = autograd.grad(orbs.sum(dim=sum_dim), grid_coords, grad_outputs=v, retain_graph=True)[0].unsqueeze(-1)
    #     time3 = time.time()
    #     print("Time for handling coordinate = ", time1 - time0)
    #     print("Time for calculating on grid orbitals = ", time2 - time1)
    #     print("Time for calculating first derivative = ", time3 - time2)
    #     if deriv == 0: return orbs
    #     if deriv == 1: return orbs, orbs_1


    # def on_grid_w_torch(self, grid_coords, deriv=0, device='cuda'):
    #     from solid_harms_sm import solid_harm
    #     from symbolic_math import eval_sympy_mat, transform_expr
    #     if 'sm_mat_g' not in self.params_list[0]: self.generate_gaussian_sm_expr()
    #     time0 = time.time()
    #     if isinstance(grid_coords, np.ndarray):
    #         grid_coords = torch.from_numpy(grid_coords).float().to(device)
    #     x, y, z = torch.split(grid_coords, 1, dim=-1)
    #     x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)
    #     if type(deriv)==int: deriv = (deriv,)
    #     from scipy.special import factorial
    #     deriv_dim = int(sum(factorial(i+2)/factorial(i)/factorial(2) for i in deriv))
    #     orbs = torch.empty((self.n_orbs,deriv_dim)+x.shape, dtype=torch.float32, device=device)
    #     time1 = time.time()
    #     # fill in result gaussian by gaussian
    #     orb_i = 0
    #     for g in self.params_list:
    #         trans_dict = BasisSet.generate_trans_dict(g)
    #         # g_mat = transform_expr(g['sm_mat_g'], trans_dict) # after rotation and translation
    #         g_mat = g['sm_mat_g'] # sm_mat_g of g in the params_list is already tranformed
    #         gauss = eval_sympy_mat(g_mat, x, y, z, deriv=0, package='torch', device=device)
    #         sh = solid_harm(g['ang'], x, y, z, trans_dict, deriv=0, package='torch', device=device)
    #         # fill in result deriv by deriv
    #         deriv_i = 0
    #         if 0 in deriv:
    #             orbs.narrow(0, orb_i,self.ang_n_orbs(g['ang']))[:,deriv_i:deriv_i+1] = sh * gauss
    #             deriv_i += 1
    #         if 1 in deriv:
    #             d_gauss = eval_sympy_mat(g_mat, x, y, z, deriv=1, package='torch', device=device)
    #             d_sh = solid_harm(g['ang'], x, y, z, trans_dict, deriv=1, package='torch', device=device)
    #             print(orb_i, 'd_sh', d_sh)
    #             print(orb_i, 'gauss', gauss)
    #             print(orb_i, 'sh', sh)
    #             print(orb_i, 'd_gauss', d_gauss)
    #             d_orbs = d_sh * gauss + sh * d_gauss
    #             orbs.narrow(0, orb_i,self.ang_n_orbs(g['ang']))[:,deriv_i:deriv_i+3] = d_orbs
    #             deriv_i += 3
    #         orb_i += self.ang_n_orbs(g['ang'])
    #     time2 = time.time()
    #     print("Time for handling coordinate = ", time1 - time0)
    #     print("Time for calculating on grid orbitals = ", time2 - time1)
    #     return orbs


    def on_grid_w_torch(self, grid_coords, deriv=0, device='cuda'):
        from solid_harms_sm import solid_harm
        from symbolic_math import eval_sympy_mat, transform_expr
        if 'sm_mat' not in self.params_list[0]: self.generate_gaussian_sm_expr()
        mem = GPU_MEM if device == 'cuda' else CPU_MEM
        time0 = time.time()
        if isinstance(grid_coords, np.ndarray):
            grid_coords = torch.from_numpy(grid_coords).float().to(device)
        x, y, z = torch.split(grid_coords, 1, dim=-1)
        x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)
        if type(deriv)==int: deriv = (deriv,)
        from scipy.special import factorial
        deriv_dim = int(sum(factorial(i+2)/factorial(i)/factorial(2) for i in deriv))
        # print("orbs shape: ", (self.n_orbs,deriv_dim)+x.shape)
        orbs = torch.empty((self.n_orbs,deriv_dim)+x.shape, dtype=torch.float32, device=device)
        time1 = time.time()
        # fill in result gaussian by gaussian
        orb_i = 0
        for g in self.params_list:
            # if 'sm_mat' not in g: self.generate_gaussian_sm_expr()
            sm_mat = g['sm_mat'] # sm_mat of g in the params_list is already tranformed
            shXgauss = eval_sympy_mat(sm_mat, x, y, z, deriv=deriv, package='torch', device=device)
            shXgauss = eval_sympy_mat(sm_mat, x, y, z, deriv=deriv, package='torch', device=device)
            orbs.narrow(0, orb_i,self.ang_n_orbs(g['ang']))[:] = shXgauss
            orb_i += self.ang_n_orbs(g['ang'])
        time2 = time.time()
        # print("Time for handling coordinate = ", time1 - time0)
        # print("Time for calculating on grid orbitals = ", time2 - time1)
        return orbs


    # def on_grid_w_torch(self, grid_coords, deriv=0, device='cuda'):
    #     from symbolic_math import torchify_str
    #     if type(deriv)==int: deriv = (deriv,)
    #     if getattr(self, '_expr_str', None) is None:
    #         self.generate_expr_str()
    #     if getattr(self, '_expr_str_1deriv', None) is None and 1 in deriv:
    #         self.generate_expr_str()
    #     mem = GPU_MEM if device == 'cuda' else CPU_MEM
    #     time0 = time.time()
    #     if isinstance(grid_coords, np.ndarray):
    #         grid_coords = torch.from_numpy(grid_coords).float().to(device)
    #     x, y, z = torch.split(grid_coords, 1, dim=-1)
    #     x, y, z = x.squeeze(-1), y.squeeze(-1), z.squeeze(-1)        
    #     from scipy.special import factorial       
    #     time1 = time.time()
    #     if 0 in deriv:
    #         expr = torchify_str(self._expr_str)
    #         orbs = torch.stack(eval(expr), dim=0)
    #         orbs.unsqueeze(1)
    #     else: orbs = torch.tensor([], dtype=torch.float32, device=device)
    #     if 1 in deriv:
    #         d_orbs = torch.stack(eval(torchify_str(self._expr_str_1deriv)), dim=0)
    #         d_orbs = d_orbs.reshape((self.n_orbs,3)+d_orbs.shape[1:])
    #         orbs = torch.cat((orbs, d_orbs), 1)
    #     time2 = time.time()
    #     print("Time for handling coordinate = ", time1 - time0)
    #     print("Time for calculating on grid orbitals = ", time2 - time1)
    #     return orbs


    def on_grid_w_numpy(self, grid_coords, deriv=0):
        from solid_harms_sm import solid_harm
        from symbolic_math import eval_sympy_mat, transform_expr
        if 'sm_mat' not in self.params_list[0]: self.generate_gaussian_sm_expr()
        time0 = time.time()
        if isinstance(grid_coords, torch.Tensor):
            grid_coords = grid_coords.detach().to('cpu').numpy()
        if len(grid_coords.shape) == 1 and len(grid_coords) == 3:
            grid_coords = np.expand_dims(grid_coords, axis=0)
        x, y, z = np.split(grid_coords, 3, axis=-1)
        x, y, z = np.squeeze(x,-1), np.squeeze(y,-1), np.squeeze(z,-1)
        if type(deriv)==int: deriv = (deriv,)
        from scipy.special import factorial
        deriv_dim = int(sum(factorial(i+2)/factorial(i)/factorial(2) for i in deriv))
        # empty orbs to fill
        orbs = np.empty((self.n_orbs,deriv_dim)+x.shape)
        time1 = time.time()
        # fill in result gaussian by gaussian
        orb_i = 0
        for g in self.params_list:
            # if 'sm_mat' not in g: self.generate_gaussian_sm_expr()
            sm_mat = g['sm_mat'] # sm_mat of g in the params_list is already tranformed
            shXgauss = eval_sympy_mat(sm_mat, x, y, z, deriv=deriv, package='numpy')
            orbs[orb_i:orb_i+self.ang_n_orbs(g['ang'])] = shXgauss
            orb_i += self.ang_n_orbs(g['ang'])
        time2 = time.time()
        print("Time for handling coordinate = ", time1 - time0)
        print("Time for calculating on grid orbitals = ", time2 - time1)
        return orbs


    # def intor(bs1, bs2): # self is bs1
    #     import sympy as sm
    #     mat_list = []
    #     for g1 in bs1.params_list:
    #         if 'sm_mat' not in g1: bs1.generate_gaussian_sm_expr()
    #         assert 1 in g1['sm_mat'].shape
    #         M1 = g1['sm_mat'].T if g1['sm_mat'].shape[0] == 1 else g1['sm_mat']
    #         row = []
    #         for g2 in bs2.params_list:
    #             if 'sm_mat' not in g2: bs2.generate_gaussian_sm_expr()
    #             assert 1 in g2['sm_mat'].shape
    #             M2 = g2['sm_mat'].T if g2['sm_mat'].shape[1] == 1 else g2['sm_mat']
    #             print(M1.shape, M2.shape)
    #             print((M1 * M2).shape)
    #             row += [M1 * M2]
    #         mat_list += [row]
    #         print(sm.Matrix(mat_list).shape)
    #     M = sm.Matrix(mat_list)
    #     oo = sm.oo
    #     return M[0, 0].integrate(('x', -oo, +oo), ('y', -oo, +oo), ('z', -oo, +oo))

                 
    @staticmethod
    def generate_trans_dict(g):
        if g['rot_dict'] is None:
            rot = None
        else:
            rot_around_z, rot_z_to = True, True
            if g['rot_dict']['around_z'] is None or g['rot_dict']['around_z'] == 0.0:
                rot_around_z = False
            if g['rot_dict']['z_to'] is None or all(g['rot_dict']['z_to'][i]==0.0 for i in (0,1,2)):
                rot_z_to = False
            if not rot_around_z and not rot_z_to:
                rot = None
            else:
                from scipy.spatial.transform import Rotation
                z_vec = np.array([0.0, 0.0, 1.0])
                if rot_around_z:
                    rot_z = Rotation.from_rotvec(z_vec*g['rot_dict']['around_z'])
                if rot_z_to:
                    z_to = g['rot_dict']['z_to']
                    z_to = z_to / np.linalg.norm(z_to)
                    rot_vec = np.cross(z_vec, z_to)
                    rot_vec *= np.arcsin(np.linalg.norm(rot_vec))/np.linalg.norm(rot_vec)
                    rot_z2 = Rotation.from_rotvec(rot_vec)
                if not rot_around_z: rot = rot_z2.as_matrix()
                elif not rot_z_to: rot = rot_around_z.as_matrix()
                else: rot = (rot_z_to * rot_around_z).as_matrix()     
        if not all(g['center'][i]==0.0 for i in (0,1,2)): move = g['center']
        else: move = None            
        return {'translation': move, 'rotation': rot}

    def generate_gaussian_sm_expr(self):
        from symbolic_math import to_sympy_mat, transform_expr
        from solid_harms_sm import solid_harm_sm
        import sympy as sm
        print("Generating symbolic expressions for orbitals ... ", end='')
        # symbolic variables
        x, y, z = sm.symbols('x, y, z')
        a, N = sm.symbols('a, N')
        gauss = N * sm.exp(-a * (x**2 + y**2 + z**2))
        # substitute numbers
        for g in self.params_list:
            trans_dict = BasisSet.generate_trans_dict(g) 
            g_expr = 0 # expression is concatenated from 0
            for i in range(g['n_contraction']):
                av = g['exponents'][i]
                Nv = g['front_factors'][i]
                g_expr += gauss.subs({a:av, N:Nv})
            g_expr = transform_expr(g_expr, trans_dict)
            gM = sm.Matrix([g_expr])
            g['sm_mat_g'] = gM
            g['sm_mat'] = solid_harm_sm(g['ang'], trans_dict=trans_dict) * gM
        print("Done.")    
        
    def generate_expr_str(self):
        import sympy
        # make a large str for the whole basis set for easy eval
        if 'sm_mat' not in self.params_list[0]: self.generate_gaussian_sm_expr()
        self._expr_str = ""
        self._expr_str_1deriv = ""
        for g in self.params_list:
            M = g['sm_mat']
            assert 1 in M.shape
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    self._expr_str += str(M[i,j])
                    self._expr_str += ', '
                    self._expr_str_1deriv += str(M[i,j].diff('x')) + ', '
                    self._expr_str_1deriv += str(M[i,j].diff('y')) + ', '
                    self._expr_str_1deriv += str(M[i,j].diff('z')) + ', '


    # def density_on_grid(self, dm, grid_coords, package='numpy', device='cpu'):
    #     start_time_phi = time.time()
    #     phi = self.calculate_on_grid(grid_coords, package=package, device=device)
    #     end_time_phi = time.time()
    #     if package == 'numpy':
    #         outer = np.matmul(np.expand_dims(phi,-1), np.expand_dims(phi,-2))
    #         rho = np.sum(outer * dm, axis=(-1,-2))
    #     elif package == 'torch':
    #         phi = self.on_grid_w_torch(grid_coords, device=device)
    #         # memory estimation
    #         GPU_MEM = 1.0e10
    #         rho = torch.empty(grid_coords.shape[:-1], dtype=torch.float32, device=device)
    #         mem_size_phi = phi.element_size()*phi.nelement()
    #         n_batch = int(np.ceil(mem_size_phi*len(dm) / GPU_MEM))
    #         batch_size = int(phi.shape[0] // n_batch)
    #         assert batch_size > 0, "grid first dim not enough for batching ..."
    #         dm = torch.from_numpy(dm).float().to(device)           
    #         for i in range(0, len(rho), batch_size):
    #             # outer = torch.matmul(phi.unsqueeze(-1), phi.unsqueeze(-2))
    #             # rho = torch.sum(outer * dm, dim=(-2,-1))
    #             bg, ed = i, min(i+batch_size,len(rho))
    #             # rho[bg:ed] = torch.einsum('...i, ...j, ij->...', phi[bg:ed], phi[bg:ed], dm)
    #             rho[bg:ed] = opt_einsum.contract('...i, ...j, ij->...', phi[bg:ed], phi[bg:ed], dm)        
    #     end_time_rho = time.time()
    #     print("Time for calculating phi = ", end_time_phi-start_time_phi)
    #     print("Time for calculating rho = ", end_6time_rho-end_time_phi)
    #     return rho


    def density_on_grid(self, dm, grid_coords, deriv=0, package='numpy', device='cpu'):
        assert self.n_orbs == len(dm)
        if type(deriv)==int: deriv = (deriv,)
        from scipy.special import factorial
        deriv_dim = int(sum(factorial(i+2)/factorial(i)/factorial(2) for i in deriv))
        time0 = time.time()
        if package == 'numpy':
            rho = np.empty((deriv_dim,) + grid_coords.shape[:-1])
            mem_size_phi = 8 * self.n_orbs * int(np.prod(rho.shape))
            mem = CPU_MEM
        elif package == 'torch':
            rho = torch.empty((deriv_dim,) + grid_coords.shape[:-1], dtype=torch.float32, device=device)           
            dm = torch.from_numpy(dm).float().to(device) 
            mem_size_phi = 4 * self.n_orbs * int(torch.prod(torch.tensor(rho.shape)))
            mem = GPU_MEM if device == 'cuda' else CPU_MEM
            print('mem_size_phi: ', mem_size_phi, 'mem, ', mem)
        time1 = time.time()
        # # memory estimation for batching
        # n_batch = int(np.ceil(mem_size_phi * 4  / mem)) # half mem for phi, half for einsum
        # batch_size = int(grid_coords.shape[0] // n_batch)
        # assert batch_size > 0, "grid first dim not enough for batching ..."
        # # fill in rho
        # for i in range(0, grid_coords.shape[0], batch_size):
        #     bg, ed = i, min(i+batch_size, grid_coords.shape[0])           
        #     tt0 = time.time()
        #     if package == 'numpy': phi = self.on_grid_w_numpy(grid_coords[bg:ed], deriv=deriv)
        #     if package == 'torch': phi = self.on_grid_w_torch(grid_coords[bg:ed], deriv=deriv, device=device)
        #     tt1 = time.time()
        #     rho[:, bg:ed] = opt_einsum.contract('i..., j..., ij->...', phi, phi, dm)
        #     tt2 = time.time()
        #     print("Time for calculating phi = ", tt1 - tt0)
        #     print("Time for calculating einsum = ", tt2 - tt1)      
        # time2 = time.time()
        # print("Time for calculating rho = ", time2 - time1)
        #      
        # memory estimation for batching
        n_batch_phi = int(np.ceil(mem_size_phi / (mem/2))) # half mem for phi, half for einsum
        n_batch_einsum = int(np.ceil(mem_size_phi*3 / (mem/2)))
        batch_size_phi = int(grid_coords.shape[0] // n_batch_phi)
        batch_size_einsum = int(grid_coords.shape[0] // n_batch_einsum)
        assert batch_size_einsum > 0, "grid first dim not enough for batching ..."
        # fill in rho
        for i in range(0, grid_coords.shape[0], batch_size_phi):
            bg, ed = i, min(i+batch_size_phi, grid_coords.shape[0])
            print('i, bg, ed', i, bg, ed)
            tt0 = time.time()
            if package == 'numpy': phi = self.on_grid_w_numpy(grid_coords[bg:ed], deriv=deriv)
            if package == 'torch': phi = self.on_grid_w_torch(grid_coords[bg:ed], deriv=deriv, device=device)
            print('phi', phi.shape)
            tt1 = time.time()
            for j in range(i, min(i+batch_size_phi, grid_coords.shape[0]), batch_size_einsum):
                bg, ed = j, min(j+batch_size_einsum, i+batch_size_phi, grid_coords.shape[0])
                print('i, j, bg, ed, batch_size_es', i, j, bg, ed, batch_size_einsum)
                rho[:, bg:ed] = opt_einsum.contract('i..., j..., ij->...', phi[:,:,bg-i:ed-i], phi[:,:,bg-i:ed-i], dm)
            tt2 = time.time()
            print("Time for calculating phi = ", tt1 - tt0)
            print("Time for calculating einsum = ", tt2 - tt1)      
        time2 = time.time()
        print("Time for calculating rho = ", time2 - time1)
        return rho


    def __add__(self, other):
        new_params_list = self.params_list + other.params_list
        return BasisSet(new_params_list)

    def __str__(self):
        return "Basis Set:\nNumber of Gaussians = " + str(len(self.params_list)) + "\nNumber of Orbitals = "+ str(self.n_orbs)




def test():
    
    # H1_augccpvqz = BasisSet(filepath="./basis_files/aug-cc-pvqz.1.gbs", centers=(0.0,0.0,0.18897261246258))
    # H2_augccpvqz = BasisSet(filepath="./basis_files/aug-cc-pvqz.1.gbs", centers=(0.0,0.0,-0.18897261246258))
    # #print(H_augccpvqz.params_list)
    # #print(H_augccpvqz.params_list)
    # # from molecule import Molecule
    # # mol1 = Molecule(linear_dists=[0.7], linear_atoms=['H','H'], charge=0, spin=0)
    # # mol1.grid = 3
    # HH_augccpvqz = H1_augccpvqz + H2_augccpvqz
    # # print(HH_augccpvqz.n_orbs)
    # coords = np.expand_dims(np.array((0.18897261246258,0.18897261246258,0.18897261246258)), 0)
    # # coords = np.random.randn(100,100,100,3)
    # # print(H1_augccpvqz.calculate_on_grid(coords, package='numpy', device='cuda'))
    # # print(H2_augccpvqz.calculate_on_grid(coords, package='torch', device='cuda'))
    # # from pyscf import dft
    # # pyscf_mol = pyscf.gto.M(atom='H 0 0 0.1;H 0 0 -0.1',basis='augccpvdz')
    # # ao_values = dft.numint.eval_ao(pyscf_mol, coords, deriv=0)
    # # print(HH_augccpvqz.params_list)
    # # print(HH_augccpvqz.calculate_on_grid(coords, package='numpy', device='cuda'))
    # # orbs = HH_augccpvqz.on_grid_w_torch(coords, deriv=1, device='cuda')
    # # orbs = HH_augccpvqz.on_grid_w_torch(coords, deriv=1, device='cuda')
    # # print('orbs = ', orbs)
    # coords = np.random.randn(1000, 1000, 100, 3)
    # dm = np.random.randn(HH_augccpvqz.n_orbs, HH_augccpvqz.n_orbs)
    # # print(HH_augccpvqz.density_on_grid(dm, coords, deriv=1, package='numpy', device='cpu'))
    # print(HH_augccpvqz.density_on_grid(dm, coords, deriv=(0,1), package='torch', device='cuda').shape)
    # # print(HH_augccpvqz.density_on_grid(dm, coords, deriv=1, package='torch', device='cuda').shape)

    # H1 = BasisSet(filepath="./basis_files/aug-cc-pvqz.1.gbs", atom='H', centers=(0.0,0.0,0.18897261246258))
    # H2 = BasisSet(filepath="./basis_files/aug-cc-pvqz.1.gbs", atom='H', centers=(0.0,0.0,-0.18897261246258))
    # g1 = H1.concat_gauss(package='torch')
    # g2 = H2.concat_gauss(package='torch')
    # g = H1.multiply_batch_gauss_torch(g1, g2)
    # BasisSet.integrate_polyXgauss_torch(g)

    H1 = BasisSet(filepath="./basis_files/aug-cc-pvqz.1.gbs", atom='H', centers=(0.0,0.0,0.18897261246258))
    print(H1.gauss_on_grid(coords=torch.tensor([1.0, 2.0, 3.0])) )
    

    # # print(H1_augccpvdz.params_list)
    # g1 = H1_augccpvdz.concat_gauss(package='torch')
    # print(g1['front_factors'])
    # g2 = H2_augccpvdz.concat_gauss(package='torch')
    # rc2 = torch.randn([10, 3])
    # g12 = H1_augccpvdz.multiply_batch_gauss_moving_center_torch(g1, g2, rc2)
    # print(g12['exponents'])

    # print(H1_augccpvdz.concated_gauss)
    


if __name__ == "__main__":
    test()


# function basis_at_this_position = H_augccpv6z_basis(x, y, z)

# r2 = x^2 + y^2 + z^2;
# % N = @(a, l) sqrt( (2*a/pi)^(3/2) * (4*a)^l / prod(2*l-1:-2:1) );

# s1 = 7.1991018333088119e+00 * ... 
#     ( 3.0422305944317218e-02 * exp(-1776.7756*r2) + ...
#       5.9800658192626671e-02 * exp(-254.0177*r2) + ...
#       1.0640707839742139e-01 * exp(-54.6980*r2) + ... 
#       1.7082927957059871e-01 * exp(-15.0183*r2) + ...
#       2.5470171407750158e-01 * exp(-4.9151*r2));
# s2 = 1.1052095797759152e+00 * exp(-1.7949*r2);
# s3 = 5.5167354934787804e-01 * exp(-0.7107*r2);
# s4 = 2.9236370137976786e-01 * exp(-0.3048*r2);
# s5 = 1.6140908015991806e-01 * exp(-0.1380*r2);
# s6 = 8.8721244705631738e-02 * exp(-0.0622*r2);
# s7 = 3.6329251104533132e-02 * exp(-0.0189*r2);

# p1 = 2.1142067705288348e+01 * exp(-8.6490*r2);
# p1x = x * p1;
# p1y = y * p1;
# p1z = z * p1;
# p2 = 6.6536160783960812e+00 * exp(-3.4300*r2);
# p2x = x * p2;
# p2y = y * p2;
# p2z = z * p2;
# p3 = 2.0934554503768719e+00 * exp(-1.3600*r2);
# p3x = x * p3;
# p3y = y * p3;
# p3z = z * p3;
# p4 = 6.5830331850507318e-01 * exp(-0.5390*r2);
# p4x = x * p4;
# p4y = y * p4;
# p4z = z * p4;
# p5 = 2.0747095430601728e-01 * exp(-0.2140*r2);
# p5x = x * p5;
# p5y = y * p5;
# p5z = z * p5;
# p6 = 4.8588511111910232e-02 * exp(-0.0670*r2);
# p6x = x * p6;
# p6y = y * p6;
# p6z = z * p6;

# solid_d = solid_harm(x, y, z, 'd');
# d1 = 2.2467340781538216e+01 * exp(-4.4530*r2);
# d1C20 = d1 * solid_d(1);
# d1C21 = d1 * solid_d(2);
# d1S21 = d1 * solid_d(3);
# d1C22 = d1 * solid_d(4);
# d1S22 = d1 * solid_d(5);
# d2 = 5.3343518649065151e+00 * exp(-1.9580*r2);
# d2C20 = d2 * solid_d(1);
# d2C21 = d2 * solid_d(2);
# d2S21 = d2 * solid_d(3);
# d2C22 = d2 * solid_d(4);
# d2S22 = d2 * solid_d(5);
# d3 = 1.2666743354499987e+00 * exp(-0.8610*r2);
# d3C20 = d3 * solid_d(1);
# d3C21 = d3 * solid_d(2);
# d3S21 = d3 * solid_d(3);
# d3C22 = d3 * solid_d(4);
# d3S22 = d3 * solid_d(5);
# d4 = 2.9993013701405702e-01 * exp(-0.3780*r2);
# d4C20 = d4 * solid_d(1);
# d4C21 = d4 * solid_d(2);
# d4S21 = d4 * solid_d(3);
# d4C22 = d4 * solid_d(4);
# d4S22 = d4 * solid_d(5);
# d5 = 4.3858917669497878e-02 * exp(-0.1260*r2);
# d5C20 = d5 * solid_d(1);
# d5C21 = d5 * solid_d(2);
# d5S21 = d5 * solid_d(3);
# d5C22 = d5 * solid_d(4);
# d5S22 = d5 * solid_d(5);

# solid_f = solid_harm(x, y, z, 'f');
# f1 = 3.5214224391558588e+01 * exp(-4.1000*r2);
# f1C30 = f1 * solid_f(1);
# f1C31 = f1 * solid_f(2);
# f1S31 = f1 * solid_f(3);
# f1C32 = f1 * solid_f(4);
# f1S32 = f1 * solid_f(5);
# f1C33 = f1 * solid_f(6);
# f1S33 = f1 * solid_f(7);
# f2 = 5.3876509944308113e+00 * exp(-1.7800*r2);
# f2C30 = f2 * solid_f(1);
# f2C31 = f2 * solid_f(2);
# f2S31 = f2 * solid_f(3);
# f2C32 = f2 * solid_f(4);
# f2S32 = f2 * solid_f(5);
# f2C33 = f2 * solid_f(6);
# f2S33 = f2 * solid_f(7);
# f3 = 8.2481834379962948e-01 * exp(-0.7730*r2);
# f3C30 = f3 * solid_f(1);
# f3C31 = f3 * solid_f(2);
# f3S31 = f3 * solid_f(3);
# f3C32 = f3 * solid_f(4);
# f3S32 = f3 * solid_f(5);
# f3C33 = f3 * solid_f(6);
# f3S33 = f3 * solid_f(7);
# f4 = 6.2169609503893865e-02 * exp(-0.2450*r2);
# f4C30 = f4 * solid_f(1);
# f4C31 = f4 * solid_f(2);
# f4S31 = f4 * solid_f(3);
# f4C32 = f4 * solid_f(4);
# f4S32 = f4 * solid_f(5);
# f4C33 = f4 * solid_f(6);
# f4S33 = f4 * solid_f(7);

# solid_g = solid_harm(x, y, z, 'g');
# g1 = 27.241096556636744 * exp(-3.1990*r2);
# g1C40 = g1 * solid_g(1);
# g1C41 = g1 * solid_g(2);
# g1S41 = g1 * solid_g(3);
# g1C42 = g1 * solid_g(4);
# g1S42 = g1 * solid_g(5);
# g1C43 = g1 * solid_g(6);
# g1S43 = g1 * solid_g(7);
# g1C44 = g1 * solid_g(8);
# g1S44 = g1 * solid_g(9);
# g2 = 2.417854243367008 * exp(-1.3260*r2);
# g2C40 = g2 * solid_g(1);
# g2C41 = g2 * solid_g(2);
# g2S41 = g2 * solid_g(3);
# g2C42 = g2 * solid_g(4);
# g2S42 = g2 * solid_g(5);
# g2C43 = g2 * solid_g(6);
# g2S43 = g2 * solid_g(7);
# g2C44 = g2 * solid_g(8);
# g2S44 = g2 * solid_g(9);
# g3 = 0.093933401824459 * exp(-0.4070*r2);
# g3C40 = g3 * solid_g(1);
# g3C41 = g3 * solid_g(2);
# g3S41 = g3 * solid_g(3);
# g3C42 = g3 * solid_g(4);
# g3S42 = g3 * solid_g(5);
# g3C43 = g3 * solid_g(6);
# g3S43 = g3 * solid_g(7);
# g3C44 = g3 * solid_g(8);
# g3S44 = g3 * solid_g(9);

# solid_h = solid_harm(x, y, z, 'h');
# h1 = 17.680324380269035 * exp(-2.6530*r2);
# h1C50 = h1 * solid_h(1);
# h1C51 = h1 * solid_h(2);
# h1S51 = h1 * solid_h(3);
# h1C52 = h1 * solid_h(4);
# h1S52 = h1 * solid_h(5);
# h1C53 = h1 * solid_h(6);
# h1S53 = h1 * solid_h(7);
# h1C54 = h1 * solid_h(8);
# h1S54 = h1 * solid_h(9);
# h1C55 = h1 * solid_h(10);
# h1S55 = h1 * solid_h(11);
# h2 = 0.213866806211079 * exp(-0.6820*r2);
# h2C50 = h2 * solid_h(1);
# h2C51 = h2 * solid_h(2);
# h2S51 = h2 * solid_h(3);
# h2C52 = h2 * solid_h(4);
# h2S52 = h2 * solid_h(5);
# h2C53 = h2 * solid_h(6);
# h2S53 = h2 * solid_h(7);
# h2C54 = h2 * solid_h(8);
# h2S54 = h2 * solid_h(9);
# h2C55 = h2 * solid_h(10);
# h2S55 = h2 * solid_h(11);

# basis_at_this_position = ...
#     [ s1, s2, s3, s4, s5, s6, s7, ...
#     p1x,p1y,p1z, p2x,p2y,p2z, p3x,p3y,p3z, p4x,p4y,p4z, p5x,p5y,p5z, p6x,p6y,p6z, ... 
#     d1S22,d1S21,d1C20,d1C21,d1C22, ...
#     d2S22,d2S21,d2C20,d2C21,d2C22, ...
#     d3S22,d3S21,d3C20,d3C21,d3C22, ...
#     d4S22,d4S21,d4C20,d4C21,d4C22, ...
#     d5S22,d5S21,d5C20,d5C21,d5C22, ...
#     f1S33,f1S32,f1S31,f1C30,f1C31,f1C32,f1C33, ... 
#     f2S33,f2S32,f2S31,f2C30,f2C31,f2C32,f2C33, ... 
#     f3S33,f3S32,f3S31,f3C30,f3C31,f3C32,f3C33, ... 
#     f4S33,f4S32,f4S31,f4C30,f4C31,f4C32,f4C33, ... 
#     g1S44,g1S43,g1S42,g1S41,g1C40,g1C41,g1C42,g1C43,g1C44, ... 
#     g2S44,g2S43,g2S42,g2S41,g2C40,g2C41,g2C42,g2C43,g2C44, ... 
#     g3S44,g3S43,g3S42,g3S41,g3C40,g3C41,g3C42,g3C43,g3C44, ... 
#     h1S55,h1S54,h1S53,h1S52,h1S51,h1C50,h1C51,h1C52,h1C53,h1C54,h1C55, ... 
#     h2S55,h2S54,h2S53,h2S52,h2S51,h2C50,h2C51,h2C52,h2C53,h2C54,h2C55 ];

# end

# function basis_at_this_position = H_augccpv6z_basis(x, y, z)

# r2 = x^2 + y^2 + z^2;
# % N = @(a, l) sqrt( (2*a/pi)^(3/2) * (4*a)^l / prod(2*l-1:-2:1) );

# s1 = 7.1991018333088119e+00 * ... 
#     ( 3.0422305944317218e-02 * exp(-1776.7756*r2) + ...
#       5.9800658192626671e-02 * exp(-254.0177*r2) + ...
#       1.0640707839742139e-01 * exp(-54.6980*r2) + ... 
#       1.7082927957059871e-01 * exp(-15.0183*r2) + ...
#       2.5470171407750158e-01 * exp(-4.9151*r2));
# s2 = 1.1052095797759152e+00 * exp(-1.7949*r2);
# s3 = 5.5167354934787804e-01 * exp(-0.7107*r2);
# s4 = 2.9236370137976786e-01 * exp(-0.3048*r2);
# s5 = 1.6140908015991806e-01 * exp(-0.1380*r2);
# s6 = 8.8721244705631738e-02 * exp(-0.0622*r2);
# s7 = 3.6329251104533132e-02 * exp(-0.0189*r2);

# p1 = 2.1142067705288348e+01 * exp(-8.6490*r2);
# p1x = x * p1;
# p1y = y * p1;
# p1z = z * p1;
# p2 = 6.6536160783960812e+00 * exp(-3.4300*r2);
# p2x = x * p2;
# p2y = y * p2;
# p2z = z * p2;
# p3 = 2.0934554503768719e+00 * exp(-1.3600*r2);
# p3x = x * p3;
# p3y = y * p3;
# p3z = z * p3;
# p4 = 6.5830331850507318e-01 * exp(-0.5390*r2);
# p4x = x * p4;
# p4y = y * p4;
# p4z = z * p4;
# p5 = 2.0747095430601728e-01 * exp(-0.2140*r2);
# p5x = x * p5;
# p5y = y * p5;
# p5z = z * p5;
# p6 = 4.8588511111910232e-02 * exp(-0.0670*r2);
# p6x = x * p6;
# p6y = y * p6;
# p6z = z * p6;

# solid_d = solid_harm(x, y, z, 'd');
# d1 = 2.2467340781538216e+01 * exp(-4.4530*r2);
# d1C20 = d1 * solid_d(1);
# d1C21 = d1 * solid_d(2);
# d1S21 = d1 * solid_d(3);
# d1C22 = d1 * solid_d(4);
# d1S22 = d1 * solid_d(5);
# d2 = 5.3343518649065151e+00 * exp(-1.9580*r2);
# d2C20 = d2 * solid_d(1);
# d2C21 = d2 * solid_d(2);
# d2S21 = d2 * solid_d(3);
# d2C22 = d2 * solid_d(4);
# d2S22 = d2 * solid_d(5);
# d3 = 1.2666743354499987e+00 * exp(-0.8610*r2);
# d3C20 = d3 * solid_d(1);
# d3C21 = d3 * solid_d(2);
# d3S21 = d3 * solid_d(3);
# d3C22 = d3 * solid_d(4);
# d3S22 = d3 * solid_d(5);
# d4 = 2.9993013701405702e-01 * exp(-0.3780*r2);
# d4C20 = d4 * solid_d(1);
# d4C21 = d4 * solid_d(2);
# d4S21 = d4 * solid_d(3);
# d4C22 = d4 * solid_d(4);
# d4S22 = d4 * solid_d(5);
# d5 = 4.3858917669497878e-02 * exp(-0.1260*r2);
# d5C20 = d5 * solid_d(1);
# d5C21 = d5 * solid_d(2);
# d5S21 = d5 * solid_d(3);
# d5C22 = d5 * solid_d(4);
# d5S22 = d5 * solid_d(5);

# solid_f = solid_harm(x, y, z, 'f');
# f1 = 3.5214224391558588e+01 * exp(-4.1000*r2);
# f1C30 = f1 * solid_f(1);
# f1C31 = f1 * solid_f(2);
# f1S31 = f1 * solid_f(3);
# f1C32 = f1 * solid_f(4);
# f1S32 = f1 * solid_f(5);
# f1C33 = f1 * solid_f(6);
# f1S33 = f1 * solid_f(7);
# f2 = 5.3876509944308113e+00 * exp(-1.7800*r2);
# f2C30 = f2 * solid_f(1);
# f2C31 = f2 * solid_f(2);
# f2S31 = f2 * solid_f(3);
# f2C32 = f2 * solid_f(4);
# f2S32 = f2 * solid_f(5);
# f2C33 = f2 * solid_f(6);
# f2S33 = f2 * solid_f(7);
# f3 = 8.2481834379962948e-01 * exp(-0.7730*r2);
# f3C30 = f3 * solid_f(1);
# f3C31 = f3 * solid_f(2);
# f3S31 = f3 * solid_f(3);
# f3C32 = f3 * solid_f(4);
# f3S32 = f3 * solid_f(5);
# f3C33 = f3 * solid_f(6);
# f3S33 = f3 * solid_f(7);
# f4 = 6.2169609503893865e-02 * exp(-0.2450*r2);
# f4C30 = f4 * solid_f(1);
# f4C31 = f4 * solid_f(2);
# f4S31 = f4 * solid_f(3);
# f4C32 = f4 * solid_f(4);
# f4S32 = f4 * solid_f(5);
# f4C33 = f4 * solid_f(6);
# f4S33 = f4 * solid_f(7);

# solid_g = solid_harm(x, y, z, 'g');
# g1 = 27.241096556636744 * exp(-3.1990*r2);
# g1C40 = g1 * solid_g(1);
# g1C41 = g1 * solid_g(2);
# g1S41 = g1 * solid_g(3);
# g1C42 = g1 * solid_g(4);
# g1S42 = g1 * solid_g(5);
# g1C43 = g1 * solid_g(6);
# g1S43 = g1 * solid_g(7);
# g1C44 = g1 * solid_g(8);
# g1S44 = g1 * solid_g(9);
# g2 = 2.417854243367008 * exp(-1.3260*r2);
# g2C40 = g2 * solid_g(1);
# g2C41 = g2 * solid_g(2);
# g2S41 = g2 * solid_g(3);
# g2C42 = g2 * solid_g(4);
# g2S42 = g2 * solid_g(5);
# g2C43 = g2 * solid_g(6);
# g2S43 = g2 * solid_g(7);
# g2C44 = g2 * solid_g(8);
# g2S44 = g2 * solid_g(9);
# g3 = 0.093933401824459 * exp(-0.4070*r2);
# g3C40 = g3 * solid_g(1);
# g3C41 = g3 * solid_g(2);
# g3S41 = g3 * solid_g(3);
# g3C42 = g3 * solid_g(4);
# g3S42 = g3 * solid_g(5);
# g3C43 = g3 * solid_g(6);
# g3S43 = g3 * solid_g(7);
# g3C44 = g3 * solid_g(8);
# g3S44 = g3 * solid_g(9);

# solid_h = solid_harm(x, y, z, 'h');
# h1 = 17.680324380269035 * exp(-2.6530*r2);
# h1C50 = h1 * solid_h(1);
# h1C51 = h1 * solid_h(2);
# h1S51 = h1 * solid_h(3);
# h1C52 = h1 * solid_h(4);
# h1S52 = h1 * solid_h(5);
# h1C53 = h1 * solid_h(6);
# h1S53 = h1 * solid_h(7);
# h1C54 = h1 * solid_h(8);
# h1S54 = h1 * solid_h(9);
# h1C55 = h1 * solid_h(10);
# h1S55 = h1 * solid_h(11);
# h2 = 0.213866806211079 * exp(-0.6820*r2);
# h2C50 = h2 * solid_h(1);
# h2C51 = h2 * solid_h(2);
# h2S51 = h2 * solid_h(3);
# h2C52 = h2 * solid_h(4);
# h2S52 = h2 * solid_h(5);
# h2C53 = h2 * solid_h(6);
# h2S53 = h2 * solid_h(7);
# h2C54 = h2 * solid_h(8);
# h2S54 = h2 * solid_h(9);
# h2C55 = h2 * solid_h(10);
# h2S55 = h2 * solid_h(11);

# basis_at_this_position = ...
#     [ s1, s2, s3, s4, s5, s6, s7, ...
#     p1x,p1y,p1z, p2x,p2y,p2z, p3x,p3y,p3z, p4x,p4y,p4z, p5x,p5y,p5z, p6x,p6y,p6z, ... 
#     d1S22,d1S21,d1C20,d1C21,d1C22, ...
#     d2S22,d2S21,d2C20,d2C21,d2C22, ...
#     d3S22,d3S21,d3C20,d3C21,d3C22, ...
#     d4S22,d4S21,d4C20,d4C21,d4C22, ...
#     d5S22,d5S21,d5C20,d5C21,d5C22, ...
#     f1S33,f1S32,f1S31,f1C30,f1C31,f1C32,f1C33, ... 
#     f2S33,f2S32,f2S31,f2C30,f2C31,f2C32,f2C33, ... 
#     f3S33,f3S32,f3S31,f3C30,f3C31,f3C32,f3C33, ... 
#     f4S33,f4S32,f4S31,f4C30,f4C31,f4C32,f4C33, ... 
#     g1S44,g1S43,g1S42,g1S41,g1C40,g1C41,g1C42,g1C43,g1C44, ... 
#     g2S44,g2S43,g2S42,g2S41,g2C40,g2C41,g2C42,g2C43,g2C44, ... 
#     g3S44,g3S43,g3S42,g3S41,g3C40,g3C41,g3C42,g3C43,g3C44, ... 
#     h1S55,h1S54,h1S53,h1S52,h1S51,h1C50,h1C51,h1C52,h1C53,h1C54,h1C55, ... 
#     h2S55,h2S54,h2S53,h2S52,h2S51,h2C50,h2C51,h2C52,h2C53,h2C54,h2C55 ];

# end