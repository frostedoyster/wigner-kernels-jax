import numpy as np
import jax
import jax.numpy as jnp

from functools import partial

from .special_functions import spherical_harmonics, scaled_spherical_bessel_i
from .dataset_processing import get_cartesian_vectors
from .spliner import compute_spline_jax


# @jax.jit
@partial(jax.vmap, in_axes=(0, None, None, 0))
def sigma(r, C_s, lambda_s, species):
    return (C_s[species[0]]+C_s[species[1]])*jnp.exp(r/(lambda_s[species[0]]+lambda_s[species[1]]))


def get_vector_expansion(vectors, radial_splines, l_max, n_max):

    r = jnp.sqrt(jnp.sum(vectors**2, axis=1))
    radial_features = compute_spline_jax(r, radial_splines["positions"], radial_splines["values"], radial_splines["derivatives"], 1)
    radial_features = radial_features.reshape((r.shape[0], l_max+1, n_max))
    angular_features = spherical_harmonics(vectors, l_max)

    angular_splits = [l**2 for l in range(1, l_max+1)]
    angular_features = jnp.split(angular_features, angular_splits, axis=1)
    radial_features = jnp.split(radial_features, l_max+1, axis=1)

    vector_expansion = []
    for l in range(l_max+1):
        vector_expansion.append(
            radial_features[l] * jnp.expand_dims(angular_features[l], -1)
        )
    return vector_expansion


def get_spherical_expansion(vector_expansion, centers, n_centers, species_neighbors, all_species_indices, n_species):

    positions_in_aggregated_vector = n_species * centers + all_species_indices[species_neighbors]

    l_max = len(vector_expansion) - 1
    spherical_expansion = []
    for l in range(l_max+1):
        n_max = vector_expansion[l].shape[2]
        spherical_expansion.append(
            jax.lax.scatter_add(
                jnp.zeros((n_centers*n_species, 2*l+1, n_max)),
                jnp.expand_dims(positions_in_aggregated_vector, -1),
                vector_expansion[l],
                jax.lax.ScatterDimensionNumbers(update_window_dims=(1, 2), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
            ).reshape((n_centers, n_species, 2*l+1, n_max)).swapaxes(1, 2).reshape((n_centers, 2*l+1, n_species*n_max))
        )
    return spherical_expansion


# @partial(jax.jit, static_argnames=["all_species", "l_max"])
def compute_wk_nu1(positions1, positions2, jax_structures1, jax_structures2, all_species, l_max, n_max, radial_splines):

    vectors1 = get_cartesian_vectors(positions1, jax_structures1)
    vectors2 = get_cartesian_vectors(positions2, jax_structures2)
    labels1 = jax_structures1["neighbor_list"]
    labels2 = jax_structures2["neighbor_list"]
    S1 = jax_structures1["structure_indices"]
    S2 = jax_structures2["structure_indices"]
    structure_offsets_1 = jax_structures1["structure_offsets"]
    structure_offsets_2 = jax_structures2["structure_offsets"]
    atomic_indices_per_element_1 = jax_structures1["atomic_indices_per_element"]
    atomic_indices_per_element_2 = jax_structures2["atomic_indices_per_element"]
    
    vector_expansion_1 = get_vector_expansion(vectors1, radial_splines, l_max, n_max)
    vector_expansion_2 = get_vector_expansion(vectors2, radial_splines, l_max, n_max)

    all_species_indices = jnp.zeros((np.max(all_species)+1,), dtype=jnp.int32)  # static computation of max
    for species_index, species in enumerate(all_species):
        all_species_indices = all_species_indices.at[species].set(species_index)
    n_centers_1 = positions1.shape[0]
    n_centers_2 = positions2.shape[0]
    n_species = len(all_species)

    Si1 = structure_offsets_1[labels1[:, 0]] + labels1[:, 1]
    Si2 = structure_offsets_2[labels2[:, 0]] + labels2[:, 1]
    
    spherical_expansion_1 = get_spherical_expansion(vector_expansion_1, Si1, n_centers_1, labels1[:, 4], all_species_indices, n_species)
    spherical_expansion_2 = get_spherical_expansion(vector_expansion_2, Si2, n_centers_2, labels2[:, 4], all_species_indices, n_species)

    species_center_1 = jax_structures1["atomic_species"]
    species_center_2 = jax_structures2["atomic_species"]

    wk_nu1_ii = {}
    s1 = {}
    s2 = {}

    for ai_idx, a_i in enumerate(all_species):

        where_ai_1 = atomic_indices_per_element_1[a_i]
        where_ai_2 = atomic_indices_per_element_2[a_i]
        s1_ai = S1[where_ai_1]
        s2_ai = S2[where_ai_2]

        wk_nu1_ii[a_i] = {}
        for l in range(l_max+1):
            wk_nu1_ii[a_i][(l, 1)] = 8.0*jnp.pi**2 * jnp.einsum("ian, jbn -> ijab", spherical_expansion_1[l][where_ai_1], spherical_expansion_2[l][where_ai_2])

        # Generate metadata for the nu=1 atom-wise kernels
        s1[a_i] = s1_ai
        s2[a_i] = s2_ai

    """for a_i in all_species:
        for l in range(l_max+1):
            print(a_i, l)
            print(wk_nu1_ii[a_i][(l, 1)][0, 0])
        exit()"""

    return wk_nu1_ii, s1, s2
