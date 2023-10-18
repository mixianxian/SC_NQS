# Copyright 2023 Tsinghua University
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiang Li <xiangxmi6@gmail.com>
#
from typing import Union
from collections import OrderedDict

import numpy as np
import jax.numpy as jnp
import jax.nn as jnn

def init_params(rng,shape,sigma=0.01):
    # sigma=0.05 in Carleo's nc work, 0.01 in NetKet default
    # https://arxiv.org/abs/1705.09792v4
    # suggests that CVNN use sigma=1/sqrt(N_input) or 1/sqrt(N_input+N_output)
    phase = 2 * rng.random(shape) * np.pi
    amplitude = rng.rayleigh(sigma,shape)
    params = amplitude * np.exp(phase*1j)
    # default np.float64
    return params

class WaveFunction:
    def __init__(self):
        pass

    def init_dict(self,n_alpha_ele:int,n_beta_ele:int,n_orb:int,n_core:int=0):
        wf_dict = {}
        wf_dict['n_alpha_ele'] = n_alpha_ele - n_core
        wf_dict['n_beta_ele'] = n_beta_ele - n_core
        wf_dict['n_orb'] = n_orb - n_core # number of space orbital
        wf_dict['n_input'] = 2 * (n_orb - n_core) # number of spin orbital or RBM input
        wf_dict['params'] = OrderedDict()
        wf_dict['params_shape'] = OrderedDict()
        wf_dict['n_params'] = 0
        return wf_dict

    def logpsi(self,wf_dict:dict,state):
        raise NotImplementedError("WaveFunction classes should implement this function")

    def Partial_logpsi(self,wf_dict:dict,state):
        raise NotImplementedError("WaveFunction classes should implement this function")
    
    def logpsi4E(self,wf_dict:dict,state):
        raise NotImplementedError("WaveFunction classes should implement this function")
    
    def logpsi_exci1e(self,wf_dict:dict,Bia,kernel,occ,vir):
        raise NotImplementedError("WaveFunction classes should implement this function")
    
    def logpsi_exci2e(self,wf_dict:dict,Bia,kernel,occ1,occ2,vir1,vir2):
        raise NotImplementedError("WaveFunction classes should implement this function")

    def update_dict(self,wf_dict:dict,updates):
        new_wf_dict = wf_dict.copy()
        new_wf_dict['params'] = wf_dict['params'].copy()
        new_wf_dict['params_shape'] = wf_dict['params_shape'].copy()
        start = 0
        for key in new_wf_dict['params'].keys():
            new_wf_dict['params'][key] += updates[start:start+np.prod(new_wf_dict['params_shape'][key])].reshape(new_wf_dict['params_shape'][key])
            start += np.prod(new_wf_dict['params_shape'][key])
        return new_wf_dict

    def save_params(self,wf_dict:dict,filename):
        np.savez(filename,**wf_dict['params'])
        params = [wf_dict['params'][key].flatten() for key in wf_dict['params'].keys()]
        params = np.hstack(params)
        np.savetxt(filename+'.txt',params,fmt='%+.5e, %+.5ej',delimiter='\n')

    def load_params_from_file(self,wf_dict:dict,filename):
        wf_dict['n_params'] = 0
        tmp = np.load(filename)
        for key in wf_dict['params'].keys():
            wf_dict['params'][key] = jnp.asarray(tmp[key])
            wf_dict['params_shape'][key] = tmp[key].shape
            wf_dict['n_params'] += np.prod(tmp[key].shape)
        tmp.close()
        print('wf_dict has been loaded.')
        self.wf_info(wf_dict)
        return wf_dict

    def wf_info(self,wf_dict:dict):
        print()
        print('Number of alpha electrons: ', wf_dict['n_alpha_ele'])
        print('Number of beta electrons: ', wf_dict['n_beta_ele'])
        print('Number of space orbitals: ', wf_dict['n_orb'])
        print('Number of spin orbitals: ', wf_dict['n_input'])
        print('Number of WaveFunction parameters: ', wf_dict['n_params'])
        print()

class RBM(WaveFunction):
    def init_dict(self,n_alpha_ele:int,n_beta_ele:int,n_orb:int,alpha:Union[int,float],sigma=None,n_core:int=0):
        wf_dict = super().init_dict(n_alpha_ele,n_beta_ele,n_orb,n_core)
        wf_dict['n_hidden'] = int(wf_dict['n_input'] * alpha)
        if sigma is None:
            sigma = 1/np.sqrt(wf_dict['n_input']+wf_dict['n_hidden'])
        rng = np.random.default_rng(42)

        wf_dict['params']['RBM_kernel'] = jnp.asarray(init_params(rng,(wf_dict['n_input'],wf_dict['n_hidden']),sigma=sigma), dtype=jnp.complex128)
        wf_dict['params_shape']['RBM_kernel'] = wf_dict['params']['RBM_kernel'].shape
        wf_dict['n_params'] += np.prod(wf_dict['params_shape']['RBM_kernel'])
        wf_dict['params']['RBM_input_bias'] = jnp.asarray(init_params(rng,(wf_dict['n_input'],),sigma=sigma), dtype=jnp.complex128)
        wf_dict['params_shape']['RBM_input_bias'] = wf_dict['params']['RBM_input_bias'].shape
        wf_dict['n_params'] += np.prod(wf_dict['params_shape']['RBM_input_bias'])
        wf_dict['params']['RBM_hidden_bias'] = jnp.asarray(init_params(rng,(wf_dict['n_hidden'],),sigma=sigma), dtype=jnp.complex128)
        wf_dict['params_shape']['RBM_hidden_bias'] = wf_dict['params']['RBM_hidden_bias'].shape
        wf_dict['n_params'] += np.prod(wf_dict['params_shape']['RBM_hidden_bias'])

        return wf_dict

    def logpsi(self,wf_dict:dict,state):
        kernel_x = wf_dict['params']['RBM_hidden_bias'] + jnp.dot(state,wf_dict['params']['RBM_kernel'])
        return jnp.dot(state,wf_dict['params']['RBM_input_bias']) + jnp.sum(jnn.softplus(kernel_x))

    def Partial_logpsi(self,wf_dict:dict,state):
        kernel_x = wf_dict['params']['RBM_hidden_bias'] + jnp.dot(state,wf_dict['params']['RBM_kernel'])
        kernel_x = jnn.sigmoid(kernel_x)
        return jnp.hstack([ jnp.dot(state[:,None],kernel_x[None,:]).flatten(), state, kernel_x ]).astype(jnp.complex128)
    
    def logpsi4E(self,wf_dict:dict,state):
        Bia = jnp.dot(state,wf_dict['params']['RBM_input_bias'])
        kernel = wf_dict['params']['RBM_hidden_bias'] + jnp.dot(state,wf_dict['params']['RBM_kernel'])
        return Bia+jnp.sum(jnn.softplus(kernel)), Bia, kernel
    
    def logpsi_exci1e(self,wf_dict:dict,Bia,kernel,occ,vir):
        new_Bia = Bia - wf_dict['params']['RBM_input_bias'][occ] + wf_dict['params']['RBM_input_bias'][vir]
        new_kernel = kernel - wf_dict['params']['RBM_kernel'][occ] + wf_dict['params']['RBM_kernel'][vir]
        return new_Bia+jnp.sum(jnn.softplus(new_kernel))
    
    def logpsi_exci2e(self,wf_dict:dict,Bia,kernel,occ1,occ2,vir1,vir2):
        new_Bia = Bia - wf_dict['params']['RBM_input_bias'][occ1] - wf_dict['params']['RBM_input_bias'][occ2]\
                      + wf_dict['params']['RBM_input_bias'][vir1] + wf_dict['params']['RBM_input_bias'][vir2]
        new_kernel = kernel - wf_dict['params']['RBM_kernel'][occ1] - wf_dict['params']['RBM_kernel'][occ2]\
                            + wf_dict['params']['RBM_kernel'][vir1] + wf_dict['params']['RBM_kernel'][vir2]
        return new_Bia+jnp.sum(jnn.softplus(new_kernel))

