# SC_NQS
Neural-network Quantum States(NQS) leverage the power of neural-network to learn the accurate ground-state wavefunctions through variational Monte Carlo approach. This repository is the implementation of "A Non-stochastic Optimization Algorithm for Neural-network Quantum States"[(https://doi.org/10.48550/arXiv.2305.12776)](https://doi.org/10.48550/arXiv.2305.12776), which depends on deterministic selection during energy evaluation and sidesteps the inherent Monte Carlo(MC) sampling process. The code is mainly developed using [JAX](https://github.com/google/jax) and utilizes [PySCF](https://github.com/pyscf/pyscf) to access electron integration of molecules.

![Table of Contents](./figures/TOC.png)

# Getting Started

1. Install requirements.

```
pip install -r requirements.txt
```

2. Compile cython codes.

```
python setup.py build_ext --inplace
```

3. Run example.

```
python test.py
```

The output result of test.py is the ground state energy of the $\rm C_2$ molecule(sto-3g basis) at equilibrium using the SC-NQS method.

# Giving Credit
If you use this code in your work, please cite the associated papers.

1. "A Non-stochastic Optimization Algorithm for Neural-network Quantum States"[(https://doi.org/10.48550/arXiv.2305.12776)](https://doi.org/10.48550/arXiv.2305.12776)
