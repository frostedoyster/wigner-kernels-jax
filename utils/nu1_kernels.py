import numpy as np
import jax
import jax.numpy as jnp

from functools import partial

from .special_functions import spherical_harmonics, scaled_spherical_bessel_i
from .dataset_processing import get_cartesian_vectors


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def sigma(r, C_s, lambda_s):
    return C_s*jnp.exp(r/lambda_s)


@partial(jax.jit, static_argnames="all_species")
def get_equal_element_pair_labels(aiaj1, aiaj2, all_species):
    # gets all pairs of indices for which ai1=ai2 and aj1=aj2
    # slightly involved approach, but it should scale linearly and not quadratically

    same_aiaj_pairs = []
    for ai in all_species:
        for aj in all_species:
            where_aiaj_1 = aiaj1[(ai, aj)]
            where_aiaj_2 = aiaj2[(ai, aj)]
            repeat_1 = jnp.repeat(where_aiaj_1, where_aiaj_2.shape[0])
            tile_2 = jnp.tile(where_aiaj_2, where_aiaj_1.shape[0])
            same_aiaj_pairs_ai = jnp.stack((repeat_1, tile_2), axis=1)
            same_aiaj_pairs.append(same_aiaj_pairs_ai)
    
    same_aiaj_pairs = jnp.concatenate(same_aiaj_pairs, axis=0)
    return same_aiaj_pairs


@partial(jax.jit, static_argnames="l_max")
def compute_wk_nu1_iijj(vectors1, vectors2, equal_element_pair_labels, l_max, C_s, lambda_s):

    sh1 = spherical_harmonics(vectors1, l_max)
    r1 = jnp.sqrt(jnp.sum(vectors1**2, axis=1))

    sh2 = spherical_harmonics(vectors2, l_max)
    r2 = jnp.sqrt(jnp.sum(vectors2**2, axis=1))

    sh1_pairs = sh1[equal_element_pair_labels[:, 0]]
    sh2_pairs = sh2[equal_element_pair_labels[:, 1]]
    r1_pairs = r1[equal_element_pair_labels[:, 0]]
    r2_pairs = r2[equal_element_pair_labels[:, 1]]

    A = 1.0/(sigma(r1_pairs, C_s, lambda_s)**2+sigma(r2_pairs, C_s, lambda_s)**2)

    prefactors = 32.0*(jnp.pi)**3*(A/(2.0*jnp.pi))**(3/2)*jnp.exp(-A*(r2_pairs-r1_pairs)**2/2.0)
    sbessi = scaled_spherical_bessel_i(A*r1_pairs*r2_pairs, l_max)

    sh_splits = [l**2 for l in range(1, l_max+1)]
    sh1_pairs = jnp.split(sh1_pairs, sh_splits, axis=1)
    sh2_pairs = jnp.split(sh2_pairs, sh_splits, axis=1)
    sbessi = jnp.split(sbessi, l_max+1, axis=1)
    sbessi = [array.squeeze(axis=-1) for array in sbessi]

    wk_nu1_iijj = {}
    for l in range(l_max+1):
        wk_nu1_iijj[l] = (((-1)**l)/(2*l+1))*(prefactors*sbessi[l])[:, None, None] * sh1_pairs[l][:, :, None] * sh2_pairs[l][:, None, :]
        # wk_nu1_iijj[l] = (prefactors*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]
        # wk_nu1_iijj[l] = ((prefactors/(2*l+1))*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]

    return wk_nu1_iijj


@partial(jax.jit, static_argnames=["all_species", "l_max"])
def compute_wk_nu1(positions1, positions2, jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s):

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
    nl_indices_per_element_pair_1 = jax_structures1["nl_indices_per_element_pair"]
    nl_indices_per_element_pair_2 = jax_structures2["nl_indices_per_element_pair"]
    
    # Get all pairs for which the center and neighbor elements are the same:
    equal_element_pair_labels = get_equal_element_pair_labels(nl_indices_per_element_pair_1, nl_indices_per_element_pair_2, all_species)
    ai_splits = np.cumsum([np.sum([nl_indices_per_element_pair_1[(ai, aj)].shape[0]*nl_indices_per_element_pair_2[(ai, aj)].shape[0] for aj in all_species]) for ai in all_species])[:-1]
    wk_iijj = compute_wk_nu1_iijj(vectors1, vectors2, equal_element_pair_labels, l_max, C_s, lambda_s)

    wk_nu1_ii = {}
    s1 = {}
    s2 = {}

    S1_pairs = labels1[equal_element_pair_labels[:, 0], 0]
    S2_pairs = labels2[equal_element_pair_labels[:, 1], 0]
    i1_pairs = labels1[equal_element_pair_labels[:, 0], 1]
    i2_pairs = labels2[equal_element_pair_labels[:, 1], 1]

    S1_pairs = jnp.split(S1_pairs, ai_splits)
    S2_pairs = jnp.split(S2_pairs, ai_splits)
    i1_pairs = jnp.split(i1_pairs, ai_splits)
    i2_pairs = jnp.split(i2_pairs, ai_splits)

    wk_iijj = {l : jnp.split(wk_iijj[l], ai_splits) for l in range(l_max+1)}

    for ai_idx, a_i in enumerate(all_species):

        S1_ai_pairs = S1_pairs[ai_idx]
        S2_ai_pairs = S2_pairs[ai_idx]
        i1_ai_pairs = i1_pairs[ai_idx]
        i2_ai_pairs = i2_pairs[ai_idx]

        where_ai_1 = atomic_indices_per_element_1[a_i]
        where_ai_2 = atomic_indices_per_element_2[a_i]
        n_i_ai_1 = where_ai_1.shape[0]
        n_i_ai_2 = where_ai_2.shape[0]
        s1_ai = S1[where_ai_1]
        s2_ai = S2[where_ai_2]

        ij_to_i_1 = structure_offsets_1[S1_ai_pairs] + i1_ai_pairs
        ij_to_i_2 = structure_offsets_2[S2_ai_pairs] + i2_ai_pairs
        indices_iijj_to_ii = n_i_ai_2*ij_to_i_1+ij_to_i_2

        wk_nu1_ii[a_i] = {}
        for l in range(l_max+1):
            wk_nu1_ii[a_i][(l, 1)] = jax.lax.scatter_add(
                jnp.zeros((n_i_ai_1*n_i_ai_2, 2*l+1, 2*l+1)),
                jnp.expand_dims(indices_iijj_to_ii, -1),
                wk_iijj[l][ai_idx],
                jax.lax.ScatterDimensionNumbers(update_window_dims=(1, 2), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
            ).reshape((n_i_ai_1, n_i_ai_2, 2*l+1, 2*l+1))

            """
            # To illustrate the very weird functioning of jax.lax.scatter_add
            import jax
            import jax.numpy as jnp
            import numpy as np

            A = jnp.array(np.random.rand(4, 2, 2))
            print(A)
            indices = jnp.array([1, 0, 1, 1])
            B = jax.lax.scatter_add(
                jnp.zeros((2, 2, 2)),
                jnp.expand_dims(indices, -1), 
                A,
                jax.lax.ScatterDimensionNumbers(update_window_dims=(1, 2), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
            )
            print()
            print(B[0]-A[1])
            print(B[1]-A[0]-A[2]-A[3])
            """

        # Generate metadata for the nu=1 atom-wise kernels
        s1[a_i] = s1_ai
        s2[a_i] = s2_ai

    return wk_nu1_ii, s1, s2
