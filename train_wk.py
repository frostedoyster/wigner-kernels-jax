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
from utils.wigner_iterations import compute_wks_high_order

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

invariant_wks = compute_wks_high_order(wks_nu1, all_species, nu_max, l_max, cgs)  # computes from nu = 1 to nu_max
invariant_wks = [wks_nu0] + invariant_wks  # prepend nu = 0

invariant_wks_per_structure = []
for nu in range(nu_max+1):
    invariant_wks_per_structure.append(
        compute_stucture_wise_kernels(invariant_wks[nu], s1, s2)
    )

print()
