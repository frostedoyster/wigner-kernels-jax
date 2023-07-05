import jax
import jax.numpy as jnp

from functools import partial

from .special_functions import spherical_harmonics, scaled_spherical_bessel_i
from .dataset_processing import get_cartesian_vectors


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def sigma(r, C_s, lambda_s):
    return C_s*jnp.exp(r/lambda_s)


def get_equal_element_pair_labels(ai1, aj1, ai2, aj2, all_species):
    # gets all pairs of indices for which ai1=ai2 and aj1=aj2
    # slightly involved approach, but it should scale linearly and not quadratically

    is_ai1 = {}
    is_aj1 = {}
    is_ai2 = {}
    is_aj2 = {}
    for a in all_species:
        is_ai1[a] = (ai1 == a)
        is_aj1[a] = (aj1 == a)
        is_ai2[a] = (ai2 == a)
        is_aj2[a] = (aj2 == a)

    same_aiaj_pairs = []
    ai_splits = {}
    index_now = 0
    for ai in all_species:
        for aj in all_species:
            where_aiaj_1 = jnp.nonzero(jnp.logical_and(is_ai1[ai], is_aj1[aj]))[0]
            where_aiaj_2 = jnp.nonzero(jnp.logical_and(is_ai2[ai], is_aj2[aj]))[0]
            repeat_1 = jnp.repeat(where_aiaj_1, where_aiaj_2.shape[0])
            tile_2 = jnp.tile(where_aiaj_2, where_aiaj_1.shape[0])
            same_aiaj_pairs_ai = jnp.stack((repeat_1, tile_2), axis=1)
            same_aiaj_pairs.append(same_aiaj_pairs_ai)
            ai_splits[ai] = (index_now, index_now+same_aiaj_pairs_ai.shape[0])
            index_now += same_aiaj_pairs_ai.shape[0]
    
    same_aiaj_pairs = jnp.concatenate(same_aiaj_pairs, axis=0)
    return same_aiaj_pairs, ai_splits


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


def compute_wk_nu1(positions1, positions2, jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s):

    vectors1 = get_cartesian_vectors(positions1, jax_structures1)
    vectors2 = get_cartesian_vectors(positions2, jax_structures2)
    labels1 = jax_structures1["neighbor_list"]
    labels2 = jax_structures2["neighbor_list"]
    S1 = jax_structures1["structure_indices"]
    S2 = jax_structures2["structure_indices"]
    structure_offsets_1 = jax_structures1["structure_offsets"]
    structure_offsets_2 = jax_structures2["structure_offsets"]
    atomic_species_1 = jax_structures1["atomic_species"]
    atomic_species_2 = jax_structures2["atomic_species"]
    
    # Get all pairs for which the center and neighbor elements are the same:
    equal_element_pair_labels, ai_splits = get_equal_element_pair_labels(labels1[:, 3], labels1[:, 4], labels2[:, 3], labels2[:, 4], all_species)
    wk_iijj = compute_wk_nu1_iijj(vectors1, vectors2, equal_element_pair_labels, l_max, C_s, lambda_s)

    wk_nu1_ii = {}
    s1 = {}
    s2 = {}

    S1_pairs = labels1[equal_element_pair_labels[:, 0], 0]
    S2_pairs = labels2[equal_element_pair_labels[:, 1], 0]
    i1_pairs = labels1[equal_element_pair_labels[:, 0], 1]
    i2_pairs = labels2[equal_element_pair_labels[:, 1], 1]

    for a_i in all_species:

        index_ai_begin, index_ai_end = ai_splits[a_i]
        S1_ai_pairs = S1_pairs[index_ai_begin:index_ai_end]
        S2_ai_pairs = S2_pairs[index_ai_begin:index_ai_end]
        i1_ai_pairs = S1_pairs[index_ai_begin:index_ai_end]
        i2_ai_pairs = S2_pairs[index_ai_begin:index_ai_end]

        where_ai_1 = jnp.where(atomic_species_1==a_i)[0]
        where_ai_2 = jnp.where(atomic_species_2==a_i)[0]
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
                wk_iijj[l][index_ai_begin:index_ai_end],
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
