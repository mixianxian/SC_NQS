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
from functools import partial
from time import time

import jax
import jax.numpy as jnp
from jax import vmap, pmap

from WaveFunctions import WaveFunction

import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import pinvh

class Optimizer:
    def __init__(self,n_params):
        self.n_params = n_params

    def get_dtheta(self, wf: WaveFunction, wf_dict: dict, parallel_states, shaped_prob, shaped_eloc, energy, n_cpu):
        raise NotImplementedError("Optimizer classes should implement this function")



class SR_Matrix(Optimizer):
    def __init__(self, param_dict: dict, n_params: int):
        self.n_params = n_params
        self.eta = param_dict['eta']
        self.diag_correct = param_dict['diag_correct']
        self.cg = param_dict['cg']

    def calc_grad(self, wf: WaveFunction, wf_dict: dict, parallel_states, shaped_prob, shaped_eloc, energy, n_cpu):
        # save all gradient may exceed memory limit
        N_limit = 50_000
        n_per_loop = N_limit // n_cpu
        n_loop = parallel_states.shape[1] // n_per_loop
        n_remain = parallel_states.shape[1] % n_per_loop
        if n_remain == 0:
            n_remain = n_per_loop
            n_loop -= 1
        part_unique_dlogpsi = pmap(vmap(partial(wf.Partial_logpsi,wf_dict)))(parallel_states[:,:n_remain,:])
        part_unique_dlogpsi = jax.device_get(part_unique_dlogpsi).reshape(-1,wf_dict['n_params'])
        part_prob = shaped_prob[:,:n_remain].reshape(-1)
        part_eloc = shaped_eloc[:,:n_remain].reshape(-1)
        # <Oi>
        oi = jnp.dot(part_prob,part_unique_dlogpsi)
        # <Oij>
        oij = jnp.dot(part_unique_dlogpsi.conj().T,part_unique_dlogpsi*part_prob[:,None])
        # gradient
        dEi = jnp.dot(part_eloc*part_prob,part_unique_dlogpsi.conj()) - energy * oi.conj()
        for j in range(n_loop):
            part_unique_dlogpsi = pmap(vmap(partial(wf.Partial_logpsi,wf_dict)))(parallel_states[:,n_remain+j*n_per_loop:n_remain+(j+1)*n_per_loop,:])
            part_unique_dlogpsi = jax.device_get(part_unique_dlogpsi).reshape(-1,wf_dict['n_params'])
            part_prob = shaped_prob[:,n_remain+j*n_per_loop:n_remain+(j+1)*n_per_loop].reshape(-1)
            part_eloc = shaped_eloc[:,n_remain+j*n_per_loop:n_remain+(j+1)*n_per_loop].reshape(-1)
            tmp_oi = jnp.dot(part_prob,part_unique_dlogpsi)
            # <Oi>
            oi += tmp_oi
            # <Oij>
            oij += jnp.dot(part_unique_dlogpsi.conj().T,part_unique_dlogpsi*part_prob[:,None])
            # gradient
            dEi += jnp.dot(part_eloc*part_prob,part_unique_dlogpsi.conj()) - energy * tmp_oi.conj()
        # S matrix
        oij -= jnp.dot(oi.conj()[:,None],oi[None,:])
        return oij, dEi

    def get_dtheta(self, wf: WaveFunction, wf_dict: dict, parallel_states, shaped_prob, shaped_eloc, energy, n_cpu):
        start = time()
        oij, dEi = self.calc_grad(wf,wf_dict,parallel_states,shaped_prob,shaped_eloc,energy,n_cpu)
        print('The matrix building cost {:.2f} min.'.format((time()-start)/60))
        oij = np.asarray(oij)
        dEi = np.asarray(dEi)
        start = time()
        if self.cg == True:
            n_try = 1
            while n_try<=1:
                if self.diag_correct == 'const':
                    oij = oij + np.diag(np.ones(self.n_params,dtype=np.complex128)) * self.eta * 2**n_try
                elif self.diag_correct == 'relative':
                    oij = oij + np.diag(np.diag(oij)) * self.eta * 2**n_try
                #oij = sparse.csr_matrix(oij)
                vec, info = cg(oij,dEi,maxiter=self.n_params*3)
                if info == 0:
                    vec = jnp.asarray(vec)
                    break
                elif info > 0:
                    print('cg process: convergence to tolerance not achieved. {} cg processes have been done.'.format(n_try))
                elif info < 0:
                    print('cg process: illegal input or breakdown.')
                n_try += 1
        else:
            oij_inv = pinvh(oij)
            vec = np.dot(oij_inv,dEi)
            vec = jnp.asarray(vec)
        print('Solve the matrix equation cost {:.2f} min.'.format((time()-start)/60))
        return vec, jnp.asarray(dEi)
