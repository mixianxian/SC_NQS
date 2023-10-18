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
from jax import lax, random, vmap, pmap

from WaveFunctions import WaveFunction
from utils import sd_excitation, state2occ, init_states_hf

class MetroSampler:
    def __init__(self,n_alpha_ele:int,n_beta_ele:int,n_orb:int,void_move:int=-1,equi_move:int=400):
        # no core electron are included
        self.n_orb = n_orb
        self.n_alpha_ele = n_alpha_ele
        self.n_beta_ele = n_beta_ele
        self.s_excitation, self.d_excitation = sd_excitation(self.n_alpha_ele,self.n_beta_ele,self.n_orb,0)
        Ns = self.s_excitation.shape[0]
        Nd = self.d_excitation.shape[0]
        self.exci2e_prob = (jnp.sqrt(Ns**2+16*Nd)-Ns) / 8
        self.P = jnp.asarray([1-self.exci2e_prob,self.exci2e_prob])
        if void_move < 0:
            self.void_move = 10 * (self.n_alpha_ele + self.n_beta_ele)
        else:
            self.void_move = void_move
        self.equi_move = equi_move

    def ConfigMove(self,logpsi_batch_fun,carry,key):
        # On each CPU, sevaral MC chains are generated simutaneously
        keys = random.split(key,self.void_move)
        states, acceptance = carry

        def change_states(states_and_logpsis,key):
            old_states, old_logpsis = states_and_logpsis
            key, new_states = lax.scan(generate_new_state,key,old_states)
            new_logpsis = logpsi_batch_fun(new_states)
            prob = 2 * (new_logpsis.real - old_logpsis.real)
            key, subkey = random.split(key)
            rand_prob = jnp.log(random.uniform(subkey,(old_states.shape[0],1))+1e-12)
            accept = prob.reshape(-1,1) > rand_prob
            new_states = jnp.where(accept,new_states,old_states)
            new_logpsis = jnp.where(accept.flatten(),new_logpsis,old_logpsis)
            acceptance = jnp.sum(accept) / old_states.shape[0]
            return (new_states,new_logpsis), acceptance

        def generate_new_state(key,state):
            occ_state = state2occ(self.n_alpha_ele,self.n_beta_ele,self.n_orb,0,state)
            # single excitation
            key, subkey = random.split(key)
            occ, vir = occ_state[self.s_excitation[random.randint(subkey,(),0,self.s_excitation.shape[0])]]
            new_state_1e = state.at[occ].set(0)
            new_state_1e = new_state_1e.at[vir].set(1)
            # double excitation
            key, subkey = random.split(key)
            occ1, occ2, vir1, vir2 = occ_state[self.d_excitation[random.randint(subkey,(),0,self.d_excitation.shape[0])]]
            new_state_2e = state.at[occ1].set(0)
            new_state_2e = new_state_2e.at[occ2].set(0)
            new_state_2e = new_state_2e.at[vir1].set(1)
            new_state_2e = new_state_2e.at[vir2].set(1)
            # chose from the two
            new_states = jnp.stack([new_state_1e,new_state_2e])
            key, subkey = random.split(key)
            return key, new_states[random.choice(subkey,jnp.arange(2),p=self.P)]

        logpsis = logpsi_batch_fun(states)
        states_and_logpsis, new_acceptance = lax.scan(change_states,(states,logpsis),keys)
        states, logpsis = states_and_logpsis
        acceptance = jnp.mean(new_acceptance) + acceptance

        return (states,acceptance), states

    def Chain_Move(self,total_steps,logpsi_batch_fun,states,key):
        # states is the initial states of each MC chain
        key, subkey = random.split(key)
        keys = random.split(subkey,self.equi_move)
        # equilibrium
        carry, _ = lax.scan(partial(self.ConfigMove,logpsi_batch_fun),(states,0.0),keys)
        states = carry[0]
        # sampling
        key, subkey = random.split(key)
        keys = random.split(subkey,total_steps)
        carry, states_chain = lax.scan(partial(self.ConfigMove,logpsi_batch_fun),(states,0.0),keys)
        acceptance = carry[1] / total_steps

        return states_chain, acceptance

def Importance_Sampling(
        wf:WaveFunction,
        wf_dict:list,
        mysampler:MetroSampler,
        key,
        total_number:int=10_000_000,
        n_cpu:int=36,
        n_chain_per_cpu:int=100,
        ):
    n_alpha_ele = wf_dict['n_alpha_ele']
    n_beta_ele  = wf_dict['n_beta_ele']
    n_orb = wf_dict['n_orb']
    assert total_number > n_cpu * n_chain_per_cpu
    n_chain_step = total_number // (n_cpu * n_chain_per_cpu)
    total_number = n_chain_step * n_cpu * n_chain_per_cpu
    start = time()

    # initialize states
    states = init_states_hf(n_cpu,n_chain_per_cpu,n_alpha_ele,n_beta_ele,n_orb)

    # initialize wf function
    logpsi_fun = partial(wf.logpsi,wf_dict)
    logpsi_batch_fun = vmap(logpsi_fun)

    # sampling states:
    # (n_cpu,n_chain_per_cpu,n_orb) -> (n_cpu,n_chain_step,n_chain_per_cpu,n_orb)
    key, subkey = random.split(key)
    keys = random.split(subkey,n_cpu)
    states_sample, acceptance = pmap(partial(mysampler.Chain_Move,n_chain_step,logpsi_batch_fun))(states,keys)
    acceptance = jnp.mean(acceptance)
    print('Metropolis-Hasting Sampling Acceptance: {:.2%}'.format(acceptance))

    # states collect and update
    states_sample = jax.device_get(states_sample).reshape(-1,2*n_orb)
    states_sample, counts = jnp.unique(states_sample,return_counts=True,axis=0)

    print('The number of unique states of this sampling is {}'.format(states_sample.shape[0]))
    print('The number of total states of this sampling is {}'.format(jnp.sum(counts)))
    print('The sampling costs {:.2f} min.'.format((time() - start)/60))

    return states_sample, counts
