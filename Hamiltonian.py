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
import jax.numpy as jnp
from jax import lax, vmap

from utils import sd_excitation, state2occ

class Hamiltonian:
    def __init__(self,E_nuc,E_hf_ele,n_alpha_ele,n_beta_ele,n_orb,h1e,r2e,n_core=0,epsilon=1e-8):
        self.E_nuc = E_nuc
        self.E_hf_ele = E_hf_ele
        self.epsilon = epsilon
        # n_core is the number of core space orbital
        self.n_orb = n_orb - n_core
        self.n_alpha_ele = n_alpha_ele - n_core
        self.n_beta_ele = n_beta_ele - n_core
        self.n_core = n_core
        # single & double excitation space
        self.s_excitation, self.d_excitation = sd_excitation(self.n_alpha_ele,self.n_beta_ele,self.n_orb,self.n_core)
        self.state_s_exci, self.state_d_exci = sd_excitation(self.n_alpha_ele,self.n_beta_ele,self.n_orb,0)
        print('Length of S&D excitation space: ', self.s_excitation.shape[0]+self.d_excitation.shape[0])

        # (a|h|b) for each spin orbital
        self.h1e = jnp.asarray(h1e, dtype=jnp.float64)
        # <ab||ij>=<ab|ij>-<ab|ji> for each spin orbital
        self.r2e = jnp.asarray(r2e, dtype=jnp.float64)
        self.Abs2e = jnp.abs(self.r2e)
        self.Smatrix = jnp.abs(self.h1e) + jnp.einsum('ijaj->ia', self.Abs2e)

    def convert_occ(self,occ_state):
        return jnp.hstack([
            occ_state[:self.n_core+self.n_alpha_ele],
            occ_state[self.n_core+self.n_orb:2*self.n_core+self.n_orb+self.n_beta_ele],
            ])

    def H_nn(self,occ_list):
        # <\psi|H|\Psi> = \sum_occ{<i|i>} + 0.5*\sum_occ{<ij||ij>}
        inx_1,inx_2 = jnp.meshgrid(occ_list,occ_list,indexing='ij')
        return jnp.sum(self.h1e[occ_list,occ_list]) + 0.5 * jnp.sum(self.r2e[inx_1,inx_2,inx_1,inx_2])

    def H_exci1e(self,occ_list,occ,vir):
        # <\psi|H|\Psi_i^a> = <i|h|a> + \sum_occ{<ij||aj>}
        # occ, vir: int, index of the orbitals. Electron excits from orb_occ to orb_vir.
        # occ and vir are what keep in occ_state
        sign = sign_1(occ_list,occ,vir)
        return sign * (self.h1e[occ,vir] + jnp.sum(self.r2e[occ,occ_list,vir,occ_list]))

    def H_exci2e(self,occ_list,occ1,occ2,vir1,vir2):
        # <\psi|H|\Psi_ij^ab> = <ij||ab>
        sign = sign_2(occ_list,occ1,occ2,vir1,vir2)
        return self.r2e[occ1,occ2,vir1,vir2] * sign
    
    def gen_singles_doubles_with_Hnm(self,state):
        occ_state = state2occ(self.n_alpha_ele,self.n_beta_ele,self.n_orb,self.n_core,state)
        occ_list = self.convert_occ(occ_state)
        # init state with core electron represented
        init_state = jnp.concatenate([
                                    jnp.ones(self.n_core),
                                    state[:self.n_orb],
                                    jnp.ones(self.n_core),
                                    state[self.n_orb:],
                                    ]).astype(jnp.int8)
        # single excitation
        def state_exci1e(occ_list,excitation):
            occ, vir = excitation
            Hnm = self.H_exci1e(occ_list,occ,vir)
            tmp_state = init_state.at[occ].set(0)
            tmp_state = tmp_state.at[vir].set(1)
            return tmp_state, Hnm
        # double excitation
        def state_exci2e(occ_list,excitation):
            occ1, occ2, vir1, vir2 = excitation
            Hnm = self.H_exci2e(occ_list,occ1,occ2,vir1,vir2)
            tmp_state = init_state.at[occ1].set(0)
            tmp_state = tmp_state.at[occ2].set(0)
            tmp_state = tmp_state.at[vir1].set(1)
            tmp_state = tmp_state.at[vir2].set(1)
            return tmp_state, Hnm

        states_exci1e, Hnm_exci1e = vmap(state_exci1e,[None,0])(occ_list,occ_state[self.s_excitation])
        states_exci2e, Hnm_exci2e = vmap(state_exci2e,[None,0])(occ_list,occ_state[self.d_excitation])

        sd_space = jnp.vstack([states_exci1e,states_exci2e])
        sd_space = jnp.hstack([sd_space[:,self.n_core:self.n_core+self.n_orb],sd_space[:,-self.n_orb:]])

        Hs_exci = jnp.hstack([Hnm_exci1e,Hnm_exci2e])

        return sd_space, Hs_exci
    
    def approx_eloc(self,logpsi4E_fun,logpsi_exci1e_fun,logpsi_exci2e_fun,PlaceHolder,state):
        occ_state = state2occ(self.n_alpha_ele,self.n_beta_ele,self.n_orb,self.n_core,state)
        occ_state_nocore = state2occ(self.n_alpha_ele,self.n_beta_ele,self.n_orb,0,state)
        occ_list = self.convert_occ(occ_state)
        logpsi_0, Bia, kernel = logpsi4E_fun(state)
        Hnn = self.H_nn(occ_list)
        def approx_exci1e(PlaceHolder,index):
            occ, vir = occ_state[self.s_excitation[index]]
            flag = self.Smatrix[occ,vir] > self.epsilon
            return PlaceHolder, lax.cond(flag,accu_exci1e,lambda _:(0.0, -1e8+0j), index)
        def approx_exci2e(PlaceHolder,index):
            occ1, occ2, vir1, vir2 = occ_state[self.d_excitation[index]]
            flag = self.Abs2e[occ1,occ2,vir1,vir2] > self.epsilon
            return PlaceHolder, lax.cond(flag,accu_exci2e,lambda _:(0.0, -1e8+0j), index)
        def accu_exci1e(index):
            occ, vir = occ_state[self.s_excitation[index]]
            Hnm = self.H_exci1e(occ_list,occ,vir)
            occ, vir = occ_state_nocore[self.state_s_exci[index]]
            logpsi = logpsi_exci1e_fun(Bia,kernel,occ,vir)
            return Hnm, logpsi
        def accu_exci2e(index):
            occ1, occ2, vir1, vir2 = occ_state[self.d_excitation[index]]
            Hnm = self.H_exci2e(occ_list,occ1,occ2,vir1,vir2)
            occ1, occ2, vir1, vir2 = occ_state_nocore[self.state_d_exci[index]]
            logpsi = logpsi_exci2e_fun(Bia,kernel,occ1,occ2,vir1,vir2)
            return Hnm, logpsi
        _, results = lax.scan(approx_exci1e,0,jnp.arange(self.s_excitation.shape[0]))
        Hnm_exci1e, logpsi_exci1e = results
        _, results = lax.scan(approx_exci2e,0,jnp.arange(self.d_excitation.shape[0]))
        Hnm_exci2e, logpsi_exci2e = results
        local_energy = Hnn + jnp.dot(Hnm_exci1e,jnp.exp(logpsi_exci1e-logpsi_0))\
                           + jnp.dot(Hnm_exci2e,jnp.exp(logpsi_exci2e-logpsi_0))
        return PlaceHolder, (local_energy,Hnn,self.epsilon)

    def approx_eloc_conn(self,logpsi4E_fun,logpsi_exci1e_fun,logpsi_exci2e_fun,PlaceHolder,state):
        occ_state = state2occ(self.n_alpha_ele,self.n_beta_ele,self.n_orb,self.n_core,state)
        occ_state_nocore = state2occ(self.n_alpha_ele,self.n_beta_ele,self.n_orb,0,state)
        occ_list = self.convert_occ(occ_state)
        logpsi_0, Bia, kernel = logpsi4E_fun(state)
        Hnn = self.H_nn(occ_list)
        def approx_exci1e(PlaceHolder,index):
            occ, vir = occ_state[self.s_excitation[index]]
            flag = self.Smatrix[occ,vir] > self.epsilon
            return PlaceHolder, lax.cond(flag,accu_exci1e,lambda _:(0.0, -1e8+0j, state), index)
        def approx_exci2e(PlaceHolder,index):
            occ1, occ2, vir1, vir2 = occ_state[self.d_excitation[index]]
            flag = self.Abs2e[occ1,occ2,vir1,vir2] > self.epsilon
            return PlaceHolder, lax.cond(flag,accu_exci2e,lambda _:(0.0, -1e8+0j, state), index)
        def accu_exci1e(index):
            occ, vir = occ_state[self.s_excitation[index]]
            Hnm = self.H_exci1e(occ_list,occ,vir)
            occ, vir = occ_state_nocore[self.state_s_exci[index]]
            logpsi = logpsi_exci1e_fun(Bia,kernel,occ,vir)
            tmp_state = state.at[occ].set(0)
            tmp_state = tmp_state.at[vir].set(1)
            return Hnm, logpsi, tmp_state
        def accu_exci2e(index):
            occ1, occ2, vir1, vir2 = occ_state[self.d_excitation[index]]
            Hnm = self.H_exci2e(occ_list,occ1,occ2,vir1,vir2)
            occ1, occ2, vir1, vir2 = occ_state_nocore[self.state_d_exci[index]]
            logpsi = logpsi_exci2e_fun(Bia,kernel,occ1,occ2,vir1,vir2)
            tmp_state = state.at[occ1].set(0)
            tmp_state = tmp_state.at[occ2].set(0)
            tmp_state = tmp_state.at[vir1].set(1)
            tmp_state = tmp_state.at[vir2].set(1)
            return Hnm, logpsi, tmp_state
        _, results = lax.scan(approx_exci1e,0,jnp.arange(self.s_excitation.shape[0]))
        Hnm_exci1e, logpsi_exci1e, states_exci1e = results
        _, results = lax.scan(approx_exci2e,0,jnp.arange(self.d_excitation.shape[0]))
        Hnm_exci2e, logpsi_exci2e, states_exci2e = results
        states_exci = jnp.vstack([states_exci1e,states_exci2e])
        approx_logpsis = jnp.hstack([logpsi_exci1e,logpsi_exci2e])
        Hs_exci = jnp.hstack([Hnm_exci1e,Hnm_exci2e])
        local_energy = Hnn + jnp.dot(Hs_exci,jnp.exp(approx_logpsis-logpsi_0))        
        return PlaceHolder, (local_energy,Hnn,states_exci,approx_logpsis,self.epsilon)

def sign_1(occ_list,occ,vir):
    def true_fun(args):
        occ_list,occ,vir = args
        return jnp.sum(jnp.logical_and(occ<occ_list,occ_list<vir))
    def false_fun(args):
        occ_list,occ,vir = args
        return jnp.sum(jnp.logical_and(vir<occ_list,occ_list<occ))
    count = lax.cond(occ<vir,true_fun,false_fun,(occ_list,occ,vir))
    return (-1)**count

def sign_2(occ_list,occ1,occ2,vir1,vir2):
    def true_fun(args):
        indexes,occ2,vir2 = args
        return jnp.sum(jnp.logical_and(occ2<indexes,indexes<vir2))
    def false_fun(args):
        indexes,occ2,vir2 = args
        return jnp.sum(jnp.logical_and(vir2<indexes,indexes<occ2))
    sign = sign_1(occ_list,occ1,vir1) * sign_1(occ_list,occ2,vir2)
    count = lax.cond(occ2<vir2,true_fun,false_fun,(jnp.asarray([occ1,vir1]),occ2,vir2))
    return sign * (-1)**count
