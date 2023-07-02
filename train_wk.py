import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ase
from ase import io

from utils.dataset_processing import JaxStructures
from utils.nu0_kernels import compute_wk_nu0
from utils.nu1_kernels import compute_wk_nu1
from utils.structure_wise import compute_stucture_wise_kernels
from utils.clebsch_gordan import get_cg_coefficients

structures = ase.io.read("datasets/gold.xyz", ":10")
# structures = [ase.Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
jax_structures = JaxStructures(structures)
jax_structures1 = jax_structures
jax_structures2 = jax_structures

all_species_jax = jnp.sort(jnp.unique(jnp.concatenate([train_structure.numbers for train_structure in structures] + [test_structure.numbers for test_structure in structures])))
all_species = [int(atomic_number) for atomic_number in all_species_jax]
print(all_species)
nu_max = 4
l_max = 3
cgs = get_cg_coefficients(l_max)
r_cut = 15.0
C_s = 0.1
lambda_s = 1.0

wks_nu0, s1_0, s2_0 = compute_wk_nu0(jax_structures1, jax_structures2, all_species)
wks_nu1, s1, s2 = compute_wk_nu1(jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s)

for a_i in all_species:
    assert jnp.all(s1[a_i]==s1_0[a_i])
    assert jnp.all(s2[a_i]==s2_0[a_i])

invariant_wks_nu1 = {}
for a_i in all_species:
    invariant_wks_nu1[a_i] = wks_nu1[a_i][(0, 1)][:, :, 0, 0]
invariant_wks = [wks_nu0, invariant_wks_nu1]

def perform_wigner_operation(wk1, wk2, cg_tensor, l1, l2, L):
    dense_transformation_matrix = cg_tensor.reshape((2*l1+1)*(2*l2+1), 2*L+1)
    product = jnp.einsum("ijab, ijcd -> ijacbd", wk1, wk2)  # to be benchmarked against e.g. wk1[:, :, :, :, jnp.newax.WRONG.is, jnp.newaxis] * wk2[...]
    result = product.reshape((product.shape[0], product.shape[1], (2*l1+1)*(2*l2+1), (2*l1+1)*(2*l2+1)))
    result = result @ dense_transformation_matrix
    result = result.swapaxes(2, 3)
    result = result @ dense_transformation_matrix
    result = result.swapaxes(2, 3)
    return result

def iterate_wigner_kernels(wks1, wks2, cgs, l_max, full=True):

    # We assume that wks1 and wks2 have the same a_i keys: there should be no exceptions
    wks_out = {}
    for a_i, wks1_ai in wks1.items():
        wks2_ai = wks2[a_i]
        wks_out_ai = {}
        for (l1, s1), wk1_l1s1 in wks1_ai.items():
            for (l2, s2), wk2_l2s2 in wks2_ai.items():
                for L in range(abs(l2-l1), min(l_max, l1+l2)+1):
                    S = s1 * s2 * (-1)**(l1+l2+L)
                    if (not full): 
                        if L != 0 or S != 1: continue  # calculate only invariant kernels
                    if (L, S) not in wks_out_ai:
                        wks_out_ai[(L, S)] = perform_wigner_operation(wk1_l1s1, wk2_l2s2, cgs[(l1, l2, L)], l1, l2, L)
                    else:
                        wks_out_ai[(L, S)] = jnp.add(wks_out_ai[(L, S)], perform_wigner_operation(wk1_l1s1, wk2_l2s2, cgs[(l1, l2, L)], l1, l2, L))
        wks_out[a_i] = wks_out_ai
    return wks_out

equivariant_wks = [0, wks_nu1]
for nu in range(2, nu_max+1):
    print(nu)
    full = True if nu <= jnp.ceil(nu_max/2) else False
    if full:
        wks_nu = iterate_wigner_kernels(equivariant_wks[nu-1], wks_nu1, cgs, l_max, full=True)
        equivariant_wks.append(wks_nu)
    else:
        wks_nu = iterate_wigner_kernels(equivariant_wks[int(jnp.ceil(nu_max/2))], equivariant_wks[nu-int(jnp.ceil(nu_max/2))], cgs, l_max, full=True)

    invariant_wks_nu = {}
    for a_i in all_species:
        invariant_wks_nu[a_i] = wks_nu[a_i][(0, 1)][:, :, 0, 0]
    invariant_wks.append(invariant_wks_nu)  # keep only the invariant part

invariant_wks_per_structure = []
for nu in range(nu_max+1):
    invariant_wks_per_structure.append(
        compute_stucture_wise_kernels(invariant_wks[nu], s1, s2)
    )

print()
