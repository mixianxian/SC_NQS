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
import jax
import jax.numpy as jnp
from jax import vmap, jit, pmap

from pyscf import ao2mo, scf, mcscf
from pyscf import lo
from rebuild import rebuild, unique_combine

def save_mo_coeff(mol,save_name,type='RHF'):
    if type == 'RHF':
        my_hf = scf.RHF(mol).run()
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        mo_alpha = my_hf.mo_coeff
        mo_beta = my_hf.mo_coeff
    elif type == 'UHF':
        my_hf = scf.UHF(mol).set(init_guess='hcore').run()
        mo1 = my_hf.stability()[0]
        dm1 = my_hf.make_rdm1(mo1, my_hf.mo_occ)
        my_hf = my_hf.run(dm1)
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        mo_alpha, mo_beta = my_hf.mo_coeff
    elif type == 'CASSCF':
        # it is necessary to reset active space 
        my_hf = scf.RHF(mol).run()
        mycas = mcscf.CASSCF(my_hf,12,12)
        mo = mycas.sort_mo([19,20,21,22,23,24,25,26,27,28,29,30]) # 排序从1开始
        mycas.natorb = True
        mycas.canonicalization = True
        mycas.kernel(mo)
        mycas.analyze()
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        mo_alpha = mycas.mo_coeff
        mo_beta = mycas.mo_coeff
    #mo_alpha = lo.Boys(mol, mo_alpha).kernel()
    #mo_beta = lo.Boys(mol, mo_beta).kernel()
    np.savez(save_name,alpha_mo_coeff=mo_alpha,beta_mo_coeff=mo_beta)

def get_mol_info(mol,n_virtual:int=0,type='UHF'):
    n_ele = mol.nelectron
    n_spin = mol.spin # alpha-beta
    n_alpha_ele = (n_ele + n_spin) // 2
    n_beta_ele = (n_ele - n_spin) // 2
    n_orb = mol.nao-n_virtual # spacial orbital

    if type == 'RHF':
        my_hf = scf.RHF(mol).run()
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        #loc_orb = lo.Boys(mol, my_hf.mo_coeff).kernel()
        mo_alpha = my_hf.mo_coeff
        #mo_alpha = loc_orb
        mo_beta = my_hf.mo_coeff
        #mo_beta = loc_orb
    elif type == 'UHF':
        my_hf = scf.UHF(mol).set(init_guess='hcore').run()
        mo1 = my_hf.stability()[0]
        dm1 = my_hf.make_rdm1(mo1, my_hf.mo_occ)
        my_hf = my_hf.run(dm1)
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        #mo_alpha, mo_beta = lo.Boys(mol, my_hf.mo_coeff).kernel()
        mo_alpha, mo_beta = my_hf.mo_coeff
    elif type == 'CASSCF':
        # it is necessary to reset active space 
        my_hf = scf.RHF(mol).run()
        mycas = mcscf.CASSCF(my_hf,12,12)
        mo = mycas.sort_mo([19,20,21,22,23,24,25,26,27,28,29,30]) # 排序从1开始
        mycas.natorb = True
        mycas.canonicalization = True
        mycas.kernel(mo)
        mycas.analyze()
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        mo_alpha = mycas.mo_coeff
        mo_beta = mycas.mo_coeff
    else:
        if ('RHF' in type) or ('CASSCF' in type):
            my_hf = scf.RHF(mol).run()
        elif ('UHF' in type):
            my_hf = scf.UHF(mol).set(init_guess='hcore').run()
            mo1 = my_hf.stability()[0]
            dm1 = my_hf.make_rdm1(mo1, my_hf.mo_occ)
            my_hf = my_hf.run(dm1)
        ele_energy = my_hf.e_tot-mol.energy_nuc()
        tmp = np.load(type)
        mo_alpha = tmp['alpha_mo_coeff']
        mo_beta  = tmp['beta_mo_coeff']
        #mo_alpha = lo.Boys(mol, mo_alpha).kernel()
        #mo_beta = lo.Boys(mol, mo_beta).kernel()
        tmp.close()
        

    print("MO_energy:")
    print(my_hf.mo_energy)
    print("MO_occ:")
    print(my_hf.mo_occ)

    mo_coeff = np.hstack([mo_alpha[:,:n_orb], mo_beta[:,:n_orb]])

    # (a|h|b) for each spin orbital
    h1e = np.einsum("ij,jk,kl", mo_coeff.T, my_hf.get_hcore(), mo_coeff)
    h1e[:n_orb,n_orb:] = 0
    h1e[n_orb:,:n_orb] = 0

    # (ab|r|ij) electronic repulsion interaction
    eri = ao2mo.full(mol,mo_coeff)
    # <ab|ij> for each spin orbital
    h2e = rebuild(eri,n_orb)
    # <ab||ij>=<ab|ij>-<ab|ji> for each spin orbital
    h2e = h2e - h2e.transpose((0,1,3,2))

    h1e = np.ascontiguousarray(h1e)
    h2e = np.ascontiguousarray(h2e)
    return n_alpha_ele, n_beta_ele, n_orb, mol.energy_nuc(), ele_energy, h1e, h2e

def init_hf_state(n_alpha_ele:int,n_beta_ele:int,n_orb:int):
    state = jnp.zeros(2*n_orb,dtype=jnp.int8)
    state = state.at[:n_alpha_ele].set(1)
    state = state.at[n_orb:n_orb+n_beta_ele].set(1)
    return state

def init_states_hf(n_cpu:int,n_chain_per_cpu:int,n_alpha_ele:int,n_beta_ele:int,n_orb:int):
    state = init_hf_state(n_alpha_ele,n_beta_ele,n_orb)
    return jnp.tile(state,(n_cpu,n_chain_per_cpu,1))

def state2occ(n_alpha_ele:int,n_beta_ele:int,n_orb:int,n_core:int,state):
    '''
    convert config state to its occupation state
    occ_state: jnp.DeviceArray, shape=(2*n_orb+2*n_core,), dtype=jnp.int8
    occ_state[:n_core+n_alpha_ele], index of occupied alpha orbitals
    occ_state[n_core+n_alpha_ele:n_orb], index of virtual alpha orbitals
    occ_state[n_orb:n_orb+n_core+n_beta_ele], index of occupied beta orbitals
    occ_state[n_orb+n_core+n_beta_ele:], index of virtual beta orbitals
    '''
    occ_part = jnp.where(state==1,jnp.arange(2*n_orb),999).sort()
    vir_part = jnp.where(state==0,jnp.arange(2*n_orb),999).sort()
    occ_state = jnp.concatenate([
                                jnp.arange(n_core),
                                occ_part[:n_alpha_ele]+n_core,
                                vir_part[:n_orb-n_alpha_ele]+n_core,
                                jnp.arange(n_core)+n_core+n_orb,
                                occ_part[n_alpha_ele:n_alpha_ele+n_beta_ele]+2*n_core,
                                vir_part[n_orb-n_alpha_ele:2*n_orb-n_alpha_ele-n_beta_ele]+2*n_core,
                                ])

    return occ_state

def sd_excitation(n_alpha_ele:int,n_beta_ele:int,n_orb:int,n_core:int):
    assert n_alpha_ele >= 2
    assert n_beta_ele >= 2
    # single excitations
    # s_exciation, shape=(N,2)
    # occ, vir = s_excitaion[i]
    inx_1,inx_2 = jnp.meshgrid(jnp.arange(n_alpha_ele),jnp.arange(n_alpha_ele,n_orb),indexing='ij')
    alpha_1e =  jnp.stack([inx_1,inx_2],axis=2).reshape(-1,2)
    alpha_1e = alpha_1e + n_core

    inx_1,inx_2 = jnp.meshgrid(jnp.arange(n_beta_ele),jnp.arange(n_beta_ele,n_orb),indexing='ij')
    beta_1e =  jnp.stack([inx_1,inx_2],axis=2).reshape(-1,2)
    beta_1e = beta_1e + n_orb + 2*n_core

    s_excitation = jnp.vstack([alpha_1e,beta_1e])

    # double excitations with diff spin
    # d_diff_excitation, shape=(N,4)
    # occ_alpha, vir_alpha, occ_beta, vir_beta = d_diff_excitation[i]
    d_diff_excitation = concat_pairs(alpha_1e,beta_1e)
    # occ_1, occ_2, vir_1, vir_2 = d_diff_excitation[i]
    d_diff_excitation = jnp.hstack([d_diff_excitation[:,::2],d_diff_excitation[:,1::2]])

    # double excitations with same spin
    # d_same_excitation, shape=(N,4)
    # occ_1, occ_2, vir_1, vir_2 = d_same_excitation[i]
    occ_part = jnp.stack(jnp.triu_indices(n_alpha_ele,1),axis=1) + n_core
    vir_part = jnp.stack(jnp.triu_indices(n_orb-n_alpha_ele,1),axis=1) + n_alpha_ele + n_core
    d_alpha_excitation = concat_pairs(occ_part,vir_part)

    occ_part = jnp.stack(jnp.triu_indices(n_beta_ele,1),axis=1) + n_orb + 2*n_core
    vir_part = jnp.stack(jnp.triu_indices(n_orb-n_beta_ele,1),axis=1) + n_beta_ele + n_orb + 2*n_core
    d_beta_excitation = concat_pairs(occ_part,vir_part)

    d_excitation = jnp.vstack([d_alpha_excitation,d_beta_excitation,d_diff_excitation])

    return s_excitation, d_excitation

def gen_singles_doubles(state,n_alpha_ele,n_beta_ele,n_orb):
    occ_state = state2occ(n_alpha_ele,n_beta_ele,n_orb,0,state)
    # single excitation
    def state_exci1e(excitation):
        occ, vir = excitation
        tmp_state = state.at[occ].set(0)
        tmp_state = tmp_state.at[vir].set(1)
        return tmp_state
    # double excitation
    def state_exci2e(excitation):
        occ1, occ2, vir1, vir2 = excitation
        tmp_state = state.at[occ1].set(0)
        tmp_state = tmp_state.at[occ2].set(0)
        tmp_state = tmp_state.at[vir1].set(1)
        tmp_state = tmp_state.at[vir2].set(1)
        return tmp_state
    s_excitation, d_excitation = sd_excitation(n_alpha_ele,n_beta_ele,n_orb,0)
    states_exci1e = vmap(state_exci1e)(occ_state[s_excitation])
    states_exci2e = vmap(state_exci2e)(occ_state[d_excitation])
    sd_space = jnp.vstack([states_exci1e,states_exci2e])
    return sd_space

def concat_pairs(pairs_1,pairs_2):
    # pairs_1 & pairs_2: (M,2) & (N,2) -> (M*N,4)
    def concat(pair_1,pair_2):
        return jnp.concatenate([pair_1,pair_2])

    def concatenate(pair_1,pairs_2):
        return vmap(concat,[None,0])(pair_1,pairs_2)

    return vmap(concatenate,[0,None])(pairs_1,pairs_2).reshape(-1,4)

@partial(jit,static_argnums=(2,))
def patch_states(unique_states,counts,n_cpu:int):
    # (N,2*n_orb) -> (n_cpu,N_per_cpu,2*orb)
    states_per_cpu = unique_states.shape[0] // n_cpu + 1
    n_patch = n_cpu * states_per_cpu - unique_states.shape[0]
    states_to_patch = jnp.tile(unique_states[-1],(n_patch,1)).astype(jnp.int8)
    counts_to_patch = jnp.zeros(n_patch,dtype=jnp.int64)
    unique_states = jnp.vstack([unique_states,states_to_patch]).reshape(n_cpu,states_per_cpu,-1)
    counts = jnp.hstack([counts,counts_to_patch])
    return unique_states, counts

def parallel_logpsi(states,logpsi_batch_fun,n_cpu):
    counts = jnp.zeros(states.shape[0])
    parallel_states, _ = patch_states(states,counts,n_cpu)
    logpsis = pmap(logpsi_batch_fun)(parallel_states)
    logpsis = jax.device_get(logpsis).reshape(-1)
    logpsis = logpsis[:states.shape[0]]
    return logpsis

def general_unique(states,n_cpu,axis=0):
    if states.shape[0] == 0:
        return states
    patched_states, _ = patch_states(states,jnp.zeros(states.shape[0]),n_cpu)
    unique_states = pmap(partial(jnp.unique,axis=axis,size=patched_states.shape[1]))(patched_states)
    unique_states = jnp.asarray(unique_combine(np.asarray(unique_states,dtype=np.int8)),dtype=jnp.int8)
    return jnp.unique(unique_states,axis=axis)
