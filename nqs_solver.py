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
import os
from functools import partial
from time import time
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.ticker import MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

import jax
import jax.numpy as jnp
from jax import lax, random, vmap, pmap

from WaveFunctions import WaveFunction
from Hamiltonian import Hamiltonian
from Samplers import MetroSampler
from Optimizers import SR_Matrix
from utils import init_states_hf, patch_states, init_hf_state, general_unique,gen_singles_doubles
from analyse import intersection
from lr_scheduler import lr_const

def nqs_mc(
        wf:WaveFunction,
        wf_dict:dict,
        H:Hamiltonian,
        n_cpu:int=36,
        save_name:str='WFParams',
        optimizer_param_dict:dict={'eta':1e-4},
        lr_dict:dict={'lr':1e-2,},
        void_move:int=-1,
        equi_move:int=200,
        n_chain_per_cpu:int=10,
        n_optStep:int=1000,
        unique_samples:int=100000,
        init_samples:int=360000,
        max_samples:int=10_000_000,
        statistic_opt:bool=False,
        energy_threshold:float=1e-5,
        energy_cor:float=-0.5,
        n_converge:int=10,
        save_freq:int=5,
        save_num:int=3,
        ):
    n_alpha_ele = wf_dict['n_alpha_ele']
    n_beta_ele  = wf_dict['n_beta_ele']
    n_orb = wf_dict['n_orb']
    assert init_samples > n_cpu * n_chain_per_cpu

    sampler = MetroSampler(n_alpha_ele,n_beta_ele,n_orb,void_move,equi_move)
    optimizer = SR_Matrix(optimizer_param_dict,wf_dict['n_params'])


    # lr_fun: learning rate
    lr_scheduler = lr_const(lr_dict['lr'])

    energy1_history = []
    energy2_history = []
    time_history = []
    lr_history = []
    n_unique_history = []
    n_unique = init_samples*2
    best_energy = 0
    last_energy = -H.E_nuc
    opt_flag = 0
    key = random.PRNGKey(42)
    start_time = time()
    print(100*'*')

    # below is optimization loops
    for i in range(1,n_optStep+1):
        last_time = time()
        print('Optimization step {} start ...'.format(i))
        logpsi_fun = partial(wf.logpsi,wf_dict)
        logpsi4E_fun = partial(wf.logpsi4E,wf_dict)
        logpsi_exci1e_fun = partial(wf.logpsi_exci1e,wf_dict)
        logpsi_exci2e_fun = partial(wf.logpsi_exci2e,wf_dict)
        logpsi_batch_fun = vmap(logpsi_fun)

        # sampling states:
        if (n_unique < unique_samples) and (init_samples < max_samples):
            init_samples *= 2
        n_chain_step = init_samples // (n_cpu * n_chain_per_cpu)
        fun_time = time()
        states = init_states_hf(n_cpu,n_chain_per_cpu,n_alpha_ele,n_beta_ele,n_orb)
        key, subkey = random.split(key)
        keys = random.split(subkey,n_cpu)
        unique_states, acceptance = pmap(partial(sampler.Chain_Move,n_chain_step,logpsi_batch_fun))(states,keys)
        acceptance = jnp.mean(acceptance)
        print('Metropolis-Hasting Sampling Acceptance: {:.2%}'.format(acceptance))
        # states collect and update
        unique_states = jax.device_get(unique_states).reshape(-1,2*n_orb)
        unique_states, unique_counts = jnp.unique(unique_states,return_counts=True,axis=0)
        print('The number of unique states of this sampling is {}'.format(unique_states.shape[0]))
        print('The number of total states of this sampling is {}'.format(jnp.sum(unique_counts)))
        print('The sampling costs {:.2f} min.'.format((time() - fun_time)/60))
        n_unique = unique_states.shape[0]
        n_unique_history.append(n_unique)

        # calculate properties
        def scan_eloc(states):
            _, elocs = lax.scan(partial(H.approx_eloc,logpsi4E_fun,logpsi_exci1e_fun,logpsi_exci2e_fun),0,states)
            return elocs
        fun_time = time()
        parallel_states, patched_counts = patch_states(unique_states,unique_counts,n_cpu)
        parallel_logpsis = pmap(logpsi_batch_fun)(parallel_states)
        parallel_logpsis = jax.device_get(parallel_logpsis).reshape(-1)
        logpsi_max = jnp.max(parallel_logpsis.real)
        unique_eloc, _, max_Hnm = pmap(scan_eloc)(parallel_states)
        unique_eloc = jax.device_get(unique_eloc).reshape(-1)
        max_Hnm = jnp.max(jnp.abs(max_Hnm))
        print('The maximum magnitude of abandoned Hamiltonian matrix elemet: ', max_Hnm)

        total_samples = jnp.sum(patched_counts)
        probability_1 = patched_counts / total_samples
        probability_2 = jnp.exp(2*(parallel_logpsis.real-logpsi_max))
        probability_2 = probability_2.at[unique_states.shape[0]:].set(0)
        probability_2 = probability_2 / jnp.sum(probability_2)

        energy1 = jnp.dot(unique_eloc,probability_1)
        energy_2 = jnp.dot((unique_eloc*unique_eloc.conj()).real,probability_1)
        energy_error = jnp.sqrt( (energy_2-jnp.abs(energy1)**2) / (total_samples-1) )
        print('!!! The Standard Method !!!')
        print('Variance: {:.5f}'.format(energy_2-jnp.abs(energy1)**2))
        print('VMC energy: {:.8f} +/- {:.5f} a.u.'.format(energy1.real+H.E_nuc,energy_error))
        print('Imaginary part of VMC energy: {:.2e}'.format(energy1.imag))
        print('Current energy ratio: {:.5f} a.u.'.format(-total_samples*energy_error**2/energy1.real))
        # calculate energy with probability_2
        energy2 = jnp.dot(unique_eloc,probability_2)
        energy_2 = jnp.dot((unique_eloc*unique_eloc.conj()).real,probability_2)
        energy_error = jnp.sqrt((energy_2-jnp.abs(energy2)**2)/unique_states.shape[0])
        print('!!! The Logpsi-Prob Method !!!')
        print('Variance: {:.5f}'.format(energy_2-jnp.abs(energy2)**2))
        print('VMC energy: {:.8f} +/- {:.5f} a.u.'.format(energy2.real+H.E_nuc,energy_error))
        print('Imaginary part of VMC energy: {:.2e}'.format(energy2.imag))
        index = np.argsort(np.abs(probability_2))
        print('E_loc+E_nuc of 10 most probable configuration:\n',unique_eloc[index[-10:]].real+H.E_nuc)
        index = np.argsort(np.abs(unique_eloc))
        print('Largest 5 Local Energy:\n', unique_eloc[index[-5:]].real+H.E_nuc)
        print('Energy calculation costs {:.2f} min.'.format((time()-fun_time)/60))
        energy1_history.append(energy1.real+H.E_nuc)
        energy2_history.append(energy2.real+H.E_nuc)
        if statistic_opt:
            energy = energy1
            energy_history = energy1_history
            probability = probability_1
        else:
            energy = energy2
            energy_history = energy2_history
            probability = probability_2
        if energy.real < best_energy:
            best_energy = energy.real
            wf.save_params(wf_dict,save_name+'_best')
        print('The best energy: {:.8f} a.u.'.format(best_energy+H.E_nuc))

        # Optimizer: Eloc&dlogpsi -> dEi -> dtheta
        shaped_eloc = unique_eloc.reshape(n_cpu,-1)
        shaped_prob = probability.reshape(n_cpu,-1)
        d_theta, gradient = optimizer.get_dtheta(wf,wf_dict,parallel_states,shaped_prob,shaped_eloc,energy,n_cpu)

        # lr_fun: learning rate
        lr = lr_scheduler.step(i)
        lr_history.append(lr)
        print('Learning Rate of Current step is {:.2e}'.format(lr))
        time_history.append(time()-start_time)

        if i%save_freq == 0:
            wf.save_params(wf_dict,save_name+'_step{}'.format(i))
            plot1(i,energy1_history,energy2_history,n_unique_history,lr_history,time_history,save_name)
            np.savez(save_name+'_history',energy1=energy1_history,energy2=energy2_history,unique=n_unique_history,lr=lr_history,time=time_history)
            if (i > save_num*save_freq) and ((i-save_num*save_freq)%100!=0):
                try:
                    os.remove(save_name+'_step{}.npz'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_step{}.npz'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_step{}.txt'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_step{}.txt'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_energy_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_energy_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_unique_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_unique_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_lr_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_lr_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_time_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_time_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_Etime_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_Etime_step{}.png'.format(i-save_num*save_freq)+' was found.')
 
        # apply_updates
        wf_dict = wf.update_dict(wf_dict,-lr*d_theta)

        print('!!! Convergency Info !!!')
        print('This updates step cost {:.2f} min.'.format((time()-last_time)/60))
        print('Energy change: {:.8f}'.format(energy.real - last_energy))
        ratio = 0
        power = 0
        grad = jnp.abs(gradient)
        while ratio < 0.99:
            power -= 1
            if power == -9:
                break
            ratio = jnp.sum(grad>10**power)/grad.shape[0]
            print('Percent of |grad| larger than 1e{} is {:.2%}'.format(power,ratio))
        ratio = 0
        power = 0
        grad = jnp.abs(d_theta)
        while ratio < 0.99:
            power -= 1
            if power == -9:
                break
            ratio = jnp.sum(grad>10**power)/grad.shape[0]
            print('Percent of |updates| larger than 1e{} is {:.2%}'.format(power,ratio))
        print('Time: {:.2f} h, optimization step {} ended!'.format((time()-start_time)/3600,i))
        print(100*'*')

        if (np.abs(energy.real - last_energy) < energy_threshold):
            opt_flag += 1
            if opt_flag == n_converge:
                print()
                print('Energy converged!')
                print('The whole optimization costs {:.2f} h'.format((time()-start_time)/3600))
                print('The average energy of final {} steps is: {:.5f} +/- {:.5f} a.u.'\
                    .format(n_converge,np.mean(energy_history[-n_converge:]).real,np.std(energy_history[-n_converge:])/np.sqrt(n_converge-1)))
                wf.save_params(wf_dict,save_name+'_Final_step{}'.format(i))
                plot1(i,energy1_history,energy2_history,n_unique_history,lr_history,time_history,save_name)
                break
        else:
            opt_flag = 0
        if energy.real < H.E_hf_ele+energy_cor:
            break
        last_energy = energy.real
    return energy1_history, energy2_history, n_unique_history, lr_history, time_history

def plot1(n_step,energy1_history,energy2_history,n_unique_history,lr_history,time_history,save_name):
    plt.cla()
    #plt.rcParams['font.family'] = 'font.serif'
    #plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus']=False
    steps = np.arange(1,n_step+1)
    time_history = np.asarray(time_history)

    plt.plot(steps,energy1_history,color='b',label='Statistical Probability')
    plt.plot(steps,energy2_history,color='r',label='WaveFunction Probability')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Energy (Ha)',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.legend(loc='upper right')
    plt.savefig(save_name+'_energy_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,n_unique_history,'b')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Unique Count',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_unique_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,lr_history,'b')
    plt.yscale('log')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Learning Rate',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_lr_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,time_history/3600,'b')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Training Time (h)',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_time_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(time_history/3600,energy1_history,'b')
    plt.plot(time_history/3600,energy2_history,'r')
    plt.xlabel('Time (h)',fontsize=20)
    plt.ylabel('Energy (Ha)',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_Etime_step{}'.format(n_step),bbox_inches = 'tight')

def nqs_sc(
        wf:WaveFunction,
        wf_dict:dict,
        H:Hamiltonian,
        n_cpu:int=36,
        save_name:str='WFParams',
        optimizer_param_dict:dict={'eta':1e-4},
        lr_dict:dict={'lr':1e-2,},
        void_move:int=-1,
        equi_move:int=200,
        n_chain_per_cpu:int=10,
        n_optStep:int=1000,
        init_samples:int=360000,
        epsilon:float=1e-8,
        energy_threshold:float=1e-5,
        n_converge:int=10,
        save_freq:int=5,
        save_num:int=3,
        ):
    n_alpha_ele = wf_dict['n_alpha_ele']
    n_beta_ele  = wf_dict['n_beta_ele']
    n_orb = wf_dict['n_orb']

    sampler = MetroSampler(n_alpha_ele,n_beta_ele,n_orb,void_move,equi_move)

    optimizer = SR_Matrix(optimizer_param_dict,wf_dict['n_params'])

    # lr_fun: learning rate
    lr_scheduler = lr_const(lr_dict['lr'])

    energy_history = []
    lr_history = []
    time_history = []
    n_unique_history = []
    best_energy = 0
    last_energy = -H.E_nuc
    opt_flag = 0

    # initial core space
    logpsi_fun = partial(wf.logpsi,wf_dict)
    logpsi_batch_fun = vmap(logpsi_fun)
    fun_time = time()
    n_chain_step = init_samples // (n_cpu * n_chain_per_cpu)
    states = init_states_hf(n_cpu,n_chain_per_cpu,n_alpha_ele,n_beta_ele,n_orb)
    key = random.PRNGKey(42)
    keys = random.split(key,n_cpu)
    core_space, acceptance = pmap(partial(sampler.Chain_Move,n_chain_step,logpsi_batch_fun))(states,keys)
    acceptance = jnp.mean(acceptance)
    print('Metropolis-Hasting Sampling Acceptance: {:.2%}'.format(acceptance))
    # states collect and update
    core_space = jax.device_get(core_space).reshape(-1,2*n_orb)
    core_space, counts = jnp.unique(core_space,return_counts=True,axis=0)
    print('The number of unique states of this sampling is {}'.format(core_space.shape[0]))
    print('The number of total states of this sampling is {}'.format(jnp.sum(counts)))
    print('The sampling costs {:.2f} min.'.format((time() - fun_time)/60))
    # add CISD space
    hf_state = init_hf_state(n_alpha_ele,n_beta_ele,n_orb)
    cisd_space = gen_singles_doubles(hf_state,n_alpha_ele,n_beta_ele,n_orb)
    core_space = jnp.unique(jnp.vstack([hf_state,cisd_space,core_space]),axis=0)
    print('The length of initial core space: ',core_space.shape[0])

    start_time = time()
    print(100*'*')
    # below is optimization loops
    for i in range(1,n_optStep+1):
        last_time = time()
        print('Optimization step {} start ...'.format(i))
        logpsi_fun = partial(wf.logpsi,wf_dict)
        logpsi4E_fun = partial(wf.logpsi4E,wf_dict)
        logpsi_exci1e_fun = partial(wf.logpsi_exci1e,wf_dict)
        logpsi_exci2e_fun = partial(wf.logpsi_exci2e,wf_dict)
        logpsi_batch_fun = vmap(logpsi_fun)
        def scan_eloc(states):
            _, elocs = lax.scan(partial(H.approx_eloc,logpsi4E_fun,logpsi_exci1e_fun,logpsi_exci2e_fun),0,states)
            return elocs
        def scan_eloc_with_conn(states):
            _, results = lax.scan(partial(H.approx_eloc_conn,logpsi4E_fun,logpsi_exci1e_fun,logpsi_exci2e_fun),0,states)
            return results

        fun_time = time()
        patched_core_space, _ = patch_states(core_space,jnp.zeros(core_space.shape[0]),n_cpu)
        core_logpsi = pmap(logpsi_batch_fun)(patched_core_space)
        core_logpsi = jax.device_get(core_logpsi).reshape(-1)[:core_space.shape[0]]
        core_logpsi = jnp.asarray(core_logpsi)
        logpsi_max = core_logpsi[jnp.argmax(core_logpsi.real)]
        logpsi_hf = logpsi_fun(hf_state)
        print('|Psi_hf|/|Psi_max|: {:.3f}'.format(jnp.exp((logpsi_hf-logpsi_max).real)))

        # save all tmp_conn_space may exceed memory limit
        N_limit = 64000
        n_per_loop = N_limit // n_cpu
        n_loop = patched_core_space.shape[1] // n_per_loop
        n_remain = patched_core_space.shape[1] % n_per_loop
        if n_remain == 0:
            n_remain = n_per_loop
            n_loop -= 1
        core_eloc,core_Hnn,conn_space,tmp_conn_logpsi,max_Hnms = pmap(scan_eloc_with_conn)(patched_core_space[:,:n_remain,:])
        core_eloc = jax.device_get(core_eloc)
        core_Hnn  = jax.device_get(core_Hnn)
        conn_space = jax.device_get(conn_space)
        tmp_conn_logpsi = jax.device_get(tmp_conn_logpsi)
        core_eloc = jnp.asarray(core_eloc)
        core_Hnn  = jnp.asarray(core_Hnn)
        conn_space = jnp.asarray(conn_space)
        tmp_conn_logpsi = jnp.asarray(tmp_conn_logpsi)
        flag = jnp.exp(tmp_conn_logpsi.real-logpsi_max.real)>epsilon
        max_Hnm = jnp.max(jnp.abs(max_Hnms))
        # shape change to (M,N_orb)
        conn_space = conn_space[flag]
        tmp_conn_logpsi = tmp_conn_logpsi[flag]
        flag = flag[flag]
        conn_space = general_unique(conn_space,n_cpu,axis=0)
        for j in range(n_loop):
            tmp_core_eloc,tmp_core_Hnn,tmp_conn_space,tmp_conn_logpsi,max_Hnms = pmap(scan_eloc_with_conn)(patched_core_space[:,n_remain+j*n_per_loop:n_remain+(j+1)*n_per_loop,:])
            tmp_core_eloc = jax.device_get(tmp_core_eloc)
            tmp_core_Hnn  = jax.device_get(tmp_core_Hnn)
            tmp_conn_space = jax.device_get(tmp_conn_space)
            tmp_conn_logpsi = jax.device_get(tmp_conn_logpsi)
            tmp_core_eloc = jnp.asarray(tmp_core_eloc)
            tmp_core_Hnn  = jnp.asarray(tmp_core_Hnn)
            tmp_conn_space = jnp.asarray(tmp_conn_space)
            tmp_conn_logpsi = jnp.asarray(tmp_conn_logpsi)
            flag = jnp.exp(tmp_conn_logpsi.real-logpsi_max.real)>epsilon
            max_Hnm = max(max_Hnm,jnp.max(jnp.abs(max_Hnms)))
            tmp_conn_space = tmp_conn_space[flag]
            tmp_conn_logpsi = tmp_conn_logpsi[flag]
            flag = flag[flag]
            core_eloc = jnp.hstack([core_eloc,tmp_core_eloc])
            core_Hnn  = jnp.hstack([core_Hnn,tmp_core_Hnn])
            conn_space = general_unique(jnp.vstack([conn_space,tmp_conn_space]),n_cpu,axis=0)
        print('Calculate Local Energy and Connected Space of Core Space costs {:.2f} min'.format((time()-fun_time)/60))

        # selected core space
        flag = jnp.exp(core_logpsi.real-logpsi_max.real)>epsilon
        core_eloc = core_eloc.reshape(-1)[:core_space.shape[0]]
        core_eloc = core_eloc[flag]
        core_Hnn  = core_Hnn.reshape(-1)[:core_space.shape[0]]
        core_Hnn  = core_Hnn[flag]
        core_space = core_space[flag]
        core_logpsi = core_logpsi[flag]
        print('Number of selected states from old core space: ',core_space.shape[0])

        # new states from conn_space
        fun_time = time()
        if conn_space.shape[0] > 0:
            conn_space = intersection(conn_space,core_space,n_cpu,Intersection=False)
        print('The length of unique conn space: ',conn_space.shape[0])
        if conn_space.shape[0] > 0:
            # patch conn_space
            patched_conn_space, _ = patch_states(conn_space,jnp.zeros(conn_space.shape[0]),n_cpu)
            conn_logpsi = pmap(logpsi_batch_fun)(patched_conn_space)
            conn_logpsi = jax.device_get(conn_logpsi).reshape(-1)[:conn_space.shape[0]]
            conn_eloc,conn_Hnn,max_Hnms = pmap(scan_eloc)(patched_conn_space)
            max_Hnm = max(max_Hnm,jnp.max(jnp.abs(max_Hnms)))
            conn_eloc = jax.device_get(conn_eloc).reshape(-1)[:conn_space.shape[0]]
            conn_Hnn  = jax.device_get(conn_Hnn).reshape(-1)[:conn_space.shape[0]]
            print('Calculate Local Energy of Selected Conn Space costs {:.2f} min'.format((time()-fun_time)/60))
            # new core_space
            core_space = jnp.vstack([core_space,conn_space])
            core_logpsi = jnp.hstack([core_logpsi,conn_logpsi])
            core_eloc = jnp.hstack([core_eloc,conn_eloc])
            core_Hnn  = jnp.hstack([core_Hnn,conn_Hnn])
        print('The length of new core space: ',core_space.shape[0])
        print('The maximum magnitude of abandoned Hamiltonian matrix elemet: ', max_Hnm)

        prob = jnp.exp(2*(core_logpsi.real-logpsi_max.real))
        prob = prob / jnp.sum(prob)
        energy = jnp.dot(core_eloc,prob)
        energy_2 = jnp.dot((core_eloc*core_eloc.conj()).real,prob)
        energy_error = jnp.sqrt((energy_2-jnp.abs(energy)**2)/prob.shape[0])
        print('!!! The Logpsi-Prob Method !!!')
        print('Variance: {:.5f}'.format(energy_2-jnp.abs(energy)**2))
        print('VMC energy: {:.8f} +/- {:.5f} a.u.'.format(energy.real+H.E_nuc,energy_error))
        print('Imaginary part of VMC energy: {:.2e}'.format(energy.imag))
        delta_e = ((core_eloc - energy) * (core_eloc - energy).conj()).real
        print('Variance: {:.5f}'.format(jnp.dot(delta_e,prob)))
        pt_e = jnp.dot(delta_e/(energy.real-core_Hnn),prob)
        print('Perturbative correction: {:.8f}'.format(pt_e))
        print('Perturbative corrected VMC energy: {:.8f}'.format(energy.real+H.E_nuc+pt_e))
        index = np.argsort(np.abs(prob))
        print('E_loc+E_nuc of 10 most probable configuration:\n',core_eloc[index[-10:]].real+H.E_nuc)
        index = np.argsort(np.abs(core_eloc))
        print('Largest 5 Local Energy:\n', core_eloc[index[-5:]].real+H.E_nuc)

        n_unique_history.append(core_space.shape[0])
        energy_history.append(energy.real+H.E_nuc)
        if energy.real < best_energy:
            best_energy = energy.real
            wf.save_params(wf_dict,save_name+'_best')
        print('The best energy: {:.8f} a.u.'.format(best_energy+H.E_nuc))

        patched_core_space, _ = patch_states(core_space,jnp.zeros(core_space.shape[0]),n_cpu)
        n_per_cpu = core_space.shape[0] // n_cpu + 1
        n_patch = n_cpu * n_per_cpu - core_space.shape[0]
        prob = jnp.hstack([prob,jnp.zeros(n_patch,dtype=jnp.complex128)])
        core_eloc = jnp.hstack([core_eloc,jnp.zeros(n_patch,dtype=jnp.complex128)])
        shaped_eloc = core_eloc.reshape(n_cpu,-1)
        shaped_prob = prob.reshape(n_cpu,-1)
        d_theta, gradient = optimizer.get_dtheta(wf,wf_dict,patched_core_space,shaped_prob,shaped_eloc,energy,n_cpu)

        # lr_fun: learning rate
        lr = lr_scheduler.step(i)
        lr_history.append(lr)
        print('Learning Rate of Current step is {:.2e}'.format(lr))
        time_history.append(time()-start_time)

        if i%save_freq == 0:
            wf.save_params(wf_dict,save_name+'_step{}'.format(i))
            plot2(i,energy_history,n_unique_history,lr_history,time_history,save_name)
            np.savez(save_name+'_history',energy=energy_history,unique=n_unique_history,lr=lr_history,time=time_history)
            if (i > save_num*save_freq) and ((i-save_num*save_freq)%100!=0):
                try:
                    os.remove(save_name+'_step{}.npz'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_step{}.npz'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_step{}.txt'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_step{}.txt'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_energy_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_energy_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_unique_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_unique_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_lr_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_lr_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_time_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_time_step{}.png'.format(i-save_num*save_freq)+' was found.')
                try:
                    os.remove(save_name+'_Etime_step{}.png'.format(i-save_num*save_freq))
                except:
                    print('WARNING! No such file named '+save_name+'_Etime_step{}.png'.format(i-save_num*save_freq)+' was found.')
 
        # apply_updates
        wf_dict = wf.update_dict(wf_dict,-lr*d_theta)

        end_time = time()
        print('!!! Convergency Info !!!')
        print('This updates step cost {:.2f} min.'.format((end_time-last_time)/60))
        print('Energy change: {:.8f}'.format(energy.real - last_energy))
        ratio = 0
        power = 0
        grad = jnp.abs(gradient)
        while ratio < 0.99:
            power -= 1
            if power == -9:
                break
            ratio = jnp.sum(grad>10**power)/grad.shape[0]
            print('Percent of |grad| larger than 1e{} is {:.2%}'.format(power,ratio))
        ratio = 0
        power = 0
        grad = jnp.abs(d_theta)
        while ratio < 0.99:
            power -= 1
            if power == -9:
                break
            ratio = jnp.sum(grad>10**power)/grad.shape[0]
            print('Percent of |updates| larger than 1e{} is {:.2%}'.format(power,ratio))
        print('Time: {:.2f} h, optimization step {} ended!'.format((end_time-start_time)/3600,i))
        print(100*'*')

        if (np.abs(energy.real - last_energy) < energy_threshold):
            opt_flag += 1
            if opt_flag == n_converge:
                print()
                print('Energy converged!')
                print('The whole optimization costs {:.2f} h'.format((time()-start_time)/3600))
                print('The average energy of final {} steps is: {:.5f} +/- {:.5f} a.u.'\
                    .format(n_converge,np.mean(energy_history[-n_converge:]).real,np.std(energy_history[-n_converge:])/np.sqrt(n_converge-1)))
                wf.save_params(wf_dict,save_name+'_Final_step{}'.format(i))
                plot2(i,energy_history,n_unique_history,lr_history,time_history,save_name)
                break
        else:
            opt_flag = 0

        last_energy = energy.real

    

    return energy_history, n_unique_history, lr_history, time_history, opt_flag, wf_dict

def plot2(n_step,energy_history,n_unique_history,lr_history,time_history,save_name):
    plt.cla()
    #plt.rcParams['font.family'] = 'font.serif'
    #plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus']=False
    steps = np.arange(1,n_step+1)
    time_history = np.asarray(time_history)

    plt.plot(steps,energy_history,'b')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Energy (Ha)',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_energy_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,n_unique_history,'b')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Unique Count',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_unique_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,lr_history,'b')
    plt.yscale('log')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Learning Rate',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_lr_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(steps,time_history/3600,'b')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('Training Time (h)',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_time_step{}'.format(n_step),bbox_inches = 'tight')

    plt.cla()
    plt.plot(time_history/3600,energy_history,'b')
    plt.xlabel('Time (h)',fontsize=20)
    plt.ylabel('Energy (Ha)',fontsize=20)
    plt.suptitle('Energy Training Process',fontsize=20)
    plt.savefig(save_name+'_Etime_step{}'.format(n_step),bbox_inches = 'tight')

