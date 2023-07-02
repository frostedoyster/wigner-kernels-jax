import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ase
from ase import io

from utils.dataset_processing import JaxStructures
from utils.nu1_kernels import compute_wk_nu1
from utils.structure_wise import compute_stucture_wise_kernels

structures = ase.io.read("datasets/gold.xyz", ":10")
# structures = [ase.Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
jax_structures = JaxStructures(structures)
jax_structures1 = jax_structures
jax_structures2 = jax_structures

all_species_jax = jnp.sort(jnp.unique(jnp.concatenate([train_structure.numbers for train_structure in structures] + [test_structure.numbers for test_structure in structures])))
all_species = [int(atomic_number) for atomic_number in all_species_jax]
print(all_species)
l_max = 3
r_cut = 15.0
C_s = 0.1
lambda_s = 1.0

wks, s1, s2 = compute_wk_nu1(jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s)
summed_wks = compute_stucture_wise_kernels(wks, s1, s2)

print(summed_wks[(0, 1)][:6, :6, 0, 0])
