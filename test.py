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
from pyscf import gto
import os
n_cpu = 4
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_cpu}'

import jax
import jax.config
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu,cuda")
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

from main import run

# define molecule
mol = gto.M(
            verbose = 4,
            atom = "C 0 0 0; C 0 0 1.26",
            basis = 'sto-3g',
           )

kwargs_dict_mc = {
      # Hamiltonian approxmation
      'epsilon':0.0,
      # optimizer
      'optimizer_param_dict':{'eta':1e-4,'diag_correct':'const','cg':True},
      'lr_dict':{'lr':0.10},
      # sampler
      'void_move':-1,
      'equi_move':500,
      'n_chain_per_cpu':5,
      # MC sampling
      'n_optStep':200,
      'unique_samples':240000,
      'init_samples':100_000,
      'max_samples':100_000,
      'statistic_opt':False,
      # convergency and check point
      'energy_threshold':1e-5,
      'energy_cor':-0.0,
      'n_converge':10,
      'save_freq':5,
      'save_num':1,
}

kwargs_dict_sc = {
      # Hamiltonian approxmation
      'epsilon1':0,
      # optimizer
      'optimizer_param_dict':{'eta':1e-4,'diag_correct':'const','cg':True},
      'lr_dict':{'lr':0.20},
      # MC sampler
      'void_move':-1,
      'equi_move':500,
      'n_chain_per_cpu':5,
      'init_samples':100_000,
      # SC sampler
      'n_optStep':2000,
      'epsilon2':1e-5,
      # convergency and check point
      'energy_threshold':1e-7,
      'n_converge':10,
      'save_freq':5,
      'save_num':1,
}

run(
    mol,
    n_cpu=n_cpu,
    # active space
    n_core=0,
    n_virtual=0,
    # NQS model
    wf_name='RBM',
    alpha=2,
    sigma=0.05,
    param_file=None,
    # file saving
    path_name='./output/test_',
    mc_step=True,
    kwargs_dict_mc=kwargs_dict_mc,
    sc_step=True,
    kwargs_dict_sc=kwargs_dict_sc,
    hf_type='RHF',
    end_analyze=True,
    )
