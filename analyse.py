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
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.ticker import MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

import jax
import jax.numpy as jnp
from jax import lax, vmap, pmap, jit

from WaveFunctions import WaveFunction
from Hamiltonian import Hamiltonian
from utils import parallel_logpsi, patch_states, init_hf_state

def Eloc_Contribution(wf:WaveFunction,wf_dict:dict,H:Hamiltonian,states,savename,n_cpu:int=36):
    logpsi_fun = partial(wf.logpsi,wf_dict)
    logpsi_batch_fun = vmap(logpsi_fun)
    logpsi4E_fun = partial(wf.logpsi4E,wf_dict)
    logpsi_exci1e_fun = partial(wf.logpsi_exci1e,wf_dict)
    logpsi_exci2e_fun = partial(wf.logpsi_exci2e,wf_dict)

    logpsis = parallel_logpsi(states,logpsi_batch_fun,n_cpu)
    index = jnp.argsort(logpsis.real)
    index = index[::-1]
    logpsi_max = logpsis[index[0]]
    prob = jnp.exp(2*(logpsis.real-logpsi_max.real))
    prob = prob / jnp.sum(prob)
    

    def scan_eloc(states):
        _, results = lax.scan(partial(H.approx_eloc,logpsi4E_fun,logpsi_exci1e_fun,logpsi_exci2e_fun),0,states)
        return results
    # patch unique_states
    counts = jnp.zeros(states.shape[0])
    patched_states, _ = patch_states(states,counts,n_cpu)
    # calculate E_loc
    Eloc, _, _ = pmap(scan_eloc)(patched_states)
    Eloc = jax.device_get(Eloc).reshape(-1)[:states.shape[0]]
    Eloc = jnp.asarray(Eloc)

    myplot(prob[index],Eloc[index],savename)

    return prob[index],Eloc[index]

def myplot(prob,Eloc,savename):
    count = prob.shape[0]
    plt.cla()
    plt.rcParams['axes.unicode_minus']=False
    steps = np.arange(1,count+1)

    plt.plot(steps,np.abs(prob),'b')
    plt.suptitle('Distribution of Probability',fontsize=20)
    plt.yscale('log')
    plt.xlabel('Hilbert Space Index',fontsize=20)
    plt.ylabel('Probability',fontsize=20)
    plt.savefig(savename+'_Probability',bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,np.abs(Eloc),'b')
    plt.suptitle('Distribution of Local Energy',fontsize=20)
    plt.xlabel('Hilbert Space Index',fontsize=20)
    plt.ylabel('Local Energy (Ha)',fontsize=20)
    plt.savefig(savename+'_LocalEnergy',bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,np.abs(prob*Eloc.real)/np.abs(np.dot(prob,Eloc.real)),'b')
    plt.yscale('log')
    plt.suptitle('Distribution of Energy Contribution',fontsize=20)
    plt.xlabel('Hilbert Space Index',fontsize=20)
    plt.ylabel('Energy Contribution',fontsize=20)
    plt.savefig(savename+'_EnergyContribution',bbox_inches = 'tight')

    np.savez(savename+'_AnalyzeData',prob=prob,eloc=Eloc)

@partial(jit,static_argnums=(0,1,2))
def exci_order(n_alpha_ele:int,n_beta_ele:int,n_orb:int,state):
    '''
    count number of excited electrons in state
    '''
    hf_state = init_hf_state(n_alpha_ele,n_beta_ele,n_orb)
    return jnp.sum(state^hf_state)//2

def histogram1s(n_alpha_ele:int,n_beta_ele:int,n_orb:int,samples,savename):
    '''
    histogram of one sample
    '''
    n_exci = vmap(partial(exci_order,n_alpha_ele,n_beta_ele,n_orb))(samples)
    max_exci = np.max(n_exci)
    count, _ = np.histogram(n_exci,np.arange(max_exci+2))
    print(count)
    if np.min(n_exci) == 0:
        print('HF state is contained !')
    else:
        print('No HF state !')

    plt.cla()
    plt.suptitle('Histogram of Excitation Order',fontsize=20)
    plt.xlabel('Excitation Order',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    plt.bar(np.arange(max_exci+1),count)
    plt.savefig(savename+'_ExcitationOrder',bbox_inches = 'tight')

def iseq(state_1,state_2):
    return (state_1==state_2).all()
def iseq_1d(states,state_1):
    return states, vmap(iseq,in_axes=[None,0])(state_1,states).any()
def iseq_2d(states_1,states_2):
    _, results = lax.scan(iseq_1d,states_2,states_1)
    return results
pmap_check_fun = pmap(iseq_2d,axis_name=None,in_axes=(0,None))

def intersection(samples_1,samples_2,n_cpu,Intersection=False):
    '''
    find states in samples_1 that are (not) in samples_2
    '''
    counts = jnp.zeros(samples_1.shape[0])
    patched_samples, _ = patch_states(samples_1,counts,n_cpu)
    flag = pmap_check_fun(patched_samples,samples_2)
    flag = jax.device_get(flag).reshape(-1)[:samples_1.shape[0]]
    if not Intersection:
        flag = flag^True
    return samples_1[flag]
