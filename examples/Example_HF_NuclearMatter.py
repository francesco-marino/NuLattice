import numpy as np
import sys, pathlib, os
sys.path.append(str(pathlib.Path(os.path.abspath('')) / ".."))

import NuLattice.HF.hartree_fock as hf
import NuLattice.lattice as lat
import NuLattice.operators.two_body_operators as twbo
import NuLattice.operators.one_body_operators as obo
import NuLattice.operators.three_body_operators as thbo
import NuLattice.constants_NLEFT as nleftConsts

myL= 4

# 2017 paper, arXiv:1702.05177v2
bpi = 0.7
a = 1.0 / 100.0
sL = 0.08
sNL = 0.08
c0 = -0.185 / a

# case = "neutron_matter"
case = "nuclear_matter"

# Uncomment the following lines to enable the Hamiltonian
# from the 2018 paper
# # 2018 paper, arXiv:1812.10928v2
# a  = 1.0/150.0
# sNL= 0.5
# sL = 0.061
# c0 = -3.41e-7/a**3
# c3 = -1.4e-14/a**6


lattice = lat.get_lattice(myL)
nsite = len(lattice)
print("number of lattice sites =", nsite)

my_basis = lat.get_sp_basis(myL)
nstat =  len(my_basis)
print("number of single-particle states =", nstat)

tKin=obo.tKin(myL, 3, a,mass=nleftConsts.mass)
# two-body just on one site; need to multiply strength by L**3
V2sparse = twbo.shortRangeV_2body(lattice, myL, sL, sNL, c0*myL**3, spin = 2, isospin = 2, verbose = True, sites=[[2,2,2]]) 

data_type = V2sparse.dtype
print("data type of interaction", data_type)
my_cont2=twbo.sparse_to_list_2body(V2sparse, myL)
del V2sparse



W3sparse = None
# Uncomment the following lines to enable the 3-body force
# # three-body just on one site; need to multiply strength by L**3
# W3sparse = thbo.shortRangeV_3body(lattice, myL, sL, sNL, c3*myL**3, spin = 2, isospin = 2, verbose = True,max_mem=1e9, sites=[[2,2,2]])
# data_type = W3sparse.dtype
# print("data type of interaction", data_type)


# For nuclear matter:
# Diagonalize kinetic energy and make density matrix

Tkin_mat = hf.get_1body_matrix(tKin,nstat,dtype=data_type)
vals, vecs = np.linalg.eigh(Tkin_mat)



if case == "nuclear_matter":
    Tkin_mat = hf.get_1body_matrix(tKin,nstat,dtype=data_type)
elif case == "neutron_matter":
    # we need to separate neutrons and protons
    tauz = obo.tau_z(lattice, myL)
    sep_fac=10.0**5
    neut_Tkin = obo.list_to_sparse1b(tKin) - sep_fac*obo.list_to_sparse1b(tauz) # to push protons up and neutrons down
    neut_Tkin_list=obo.sparse1b_to_list(neut_Tkin)
    Tkin_mat = hf.get_1body_matrix(neut_Tkin_list,nstat,dtype=data_type)
else: 
    print("WRONG case ARGUMENT: valid choices are 'nuclear_matter' or 'neutron_matter' ")

vals, vecs = np.linalg.eigh(Tkin_mat)



# compute magic numbers / shell closures of kinetic energy
magic_nums=[]
for i in range(1,nstat):
    if vals[i]-vals[i-1] > 1.e-4:
        magic_nums.append(i)

magic_nums.append(nstat)
print(magic_nums)


print("A, rho, E/A")
a_lat = a * nleftConsts.hbarc
selected_nums=magic_nums
for aa in selected_nums:
    dens=hf.contract("pi,qi->pq", vecs[:,0:aa], np.conjugate(vecs[:,0:aa]) )
    if W3sparse is not None:
        erg=hf.HF_energy(tKin, my_cont2, W3sparse, dens, w3_sparse=True)
    else:
        erg=hf.HF_energy(tKin, my_cont2, [], dens, w3_sparse=False)
    num = np.real(np.trace(dens))
    print(f'{num},{num/(myL*a_lat)**3},{np.real(erg)/num}')


