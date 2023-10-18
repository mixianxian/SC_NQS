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
import numpy as np
from typing import Any

from jax import random
from nqs_solver import nqs_mc, nqs_sc
from Hamiltonian import Hamiltonian
from WaveFunctions import RBM
from Samplers import MetroSampler, Importance_Sampling
from utils import get_mol_info
from analyse import Eloc_Contribution, histogram1s

def run(
    mol,
    n_cpu:int=36,
    # active space
    n_core:int=0,
    n_virtual:int=0,
    # NQS model
    wf_name:str='RBM',
    alpha:int=2,
    sigma:Any=None,
    param_file:Any=None,
    # file saving
    path_name:str='/home/lixiang/RNQS/scalar/v2/',
    mc_step=True,
    kwargs_dict_mc=None,
    sc_step=True,
    kwargs_dict_sc=None,
    hf_type='RHF',
    end_analyze=True,
    ):
    # obtain SCF calculation info
    n_alpha_ele, n_beta_ele, n_orb, E_nuc, E_ele, h1e, h2e = get_mol_info(mol,n_virtual,hf_type)
    print('HF energy from PySCF',E_ele+E_nuc)

    # define wavefunction model
    if wf_name == 'RBM':
        wf = RBM()
        wf_dict = wf.init_dict(n_alpha_ele,n_beta_ele,n_orb,alpha,sigma,n_core)
    else:
        raise ValueError(wf_name + ' are not supported Wave Function Class.')
    if param_file is not None:
        wf_dict = wf.load_params_from_file(wf_dict,param_file)
    else:
        wf.wf_info(wf_dict)
    save_name = None

    # MC optimization stage
    if mc_step:
        H = Hamiltonian(E_nuc,E_ele,n_alpha_ele, n_beta_ele, n_orb, h1e, h2e, n_core, kwargs_dict_mc['epsilon'])
        save_name = path_name + '_'.join(['MC',wf_name])
        energy1_history, energy2_history, n_unique_history, lr_history, time_history = nqs_mc(
            wf,
            wf_dict,
            H,
            n_cpu,
            save_name,
            kwargs_dict_mc['optimizer_param_dict'],
            kwargs_dict_mc['lr_dict'],
            kwargs_dict_mc['void_move'],
            kwargs_dict_mc['equi_move'],
            kwargs_dict_mc['n_chain_per_cpu'],
            kwargs_dict_mc['n_optStep'],
            kwargs_dict_mc['unique_samples'],
            kwargs_dict_mc['init_samples'],
            kwargs_dict_mc['max_samples'],
            kwargs_dict_mc['statistic_opt'],
            kwargs_dict_mc['energy_threshold'],
            kwargs_dict_mc['energy_cor'],
            kwargs_dict_mc['n_converge'],
            kwargs_dict_mc['save_freq'],
            kwargs_dict_mc['save_num'],
            )
        np.savez(save_name+'_history',energy1=energy1_history,energy2=energy2_history,unique=n_unique_history,lr=lr_history,time=time_history)

    # SC optimization stage
    if sc_step:
        if mc_step:
            wf_dict = wf.load_params_from_file(wf_dict,save_name+'_best.npz')
        H = Hamiltonian(E_nuc,E_ele,n_alpha_ele, n_beta_ele, n_orb, h1e, h2e, n_core, kwargs_dict_sc['epsilon1'])
        opt_flag = 0
        n_run = 0
        while opt_flag != kwargs_dict_sc['n_converge']:
            n_run += 1
            save_name = path_name + '_'.join(['SC',wf_name])
            energy_history, n_unique_history, lr_history, time_history, opt_flag, wf_dict, = nqs_sc(
                wf,
                wf_dict,
                H,
                n_cpu,
                save_name,
                kwargs_dict_sc['optimizer_param_dict'],
                kwargs_dict_sc['lr_dict'],
                kwargs_dict_sc['void_move'],
                kwargs_dict_sc['equi_move'],
                kwargs_dict_sc['n_chain_per_cpu'],
                kwargs_dict_sc['n_optStep'],
                kwargs_dict_sc['init_samples'],
                kwargs_dict_sc['epsilon2'],
                kwargs_dict_sc['energy_threshold'],
                kwargs_dict_sc['n_converge'],
                kwargs_dict_sc['save_freq'],
                kwargs_dict_sc['save_num'],
                )
            np.savez(save_name+'_history',energy=energy_history,unique=n_unique_history,lr=lr_history,time=time_history)
            break

    # analysis
    if end_analyze:
        if mc_step or sc_step:
            wf_dict = wf.load_params_from_file(wf_dict,save_name+'_best.npz')
        H = Hamiltonian(E_nuc,E_ele,n_alpha_ele, n_beta_ele, n_orb, h1e, h2e, n_core, kwargs_dict_sc['epsilon1'])
        sampler = MetroSampler(H.n_alpha_ele,H.n_beta_ele,H.n_orb,kwargs_dict_mc['void_move'],kwargs_dict_mc['equi_move'])
        key = random.PRNGKey(42)
        MC_samples, _ = Importance_Sampling(wf,wf_dict,sampler,key,kwargs_dict_sc['init_samples'],n_cpu,kwargs_dict_mc['n_chain_per_cpu'])
        prob,Eloc = Eloc_Contribution(wf,wf_dict,H,MC_samples,path_name+'MC_samples',n_cpu)
        print('VMC energy of MC samples: ', np.dot(prob,Eloc)+H.E_nuc)
        histogram1s(H.n_alpha_ele,H.n_beta_ele,H.n_orb,MC_samples,path_name+'MC_samples')
        