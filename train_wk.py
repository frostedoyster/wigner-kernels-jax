import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ase
from ase import io
from utils.dataset_processing import JaxStructures, get_neighbor_list, get_cartesian_vectors

structures = ase.io.read("datasets/gold.xyz", ":100")
jax_structures = JaxStructures(structures)

vectors, labels = get_cartesian_vectors(jax_structures, 4.0)

print(vectors)
print(labels)

