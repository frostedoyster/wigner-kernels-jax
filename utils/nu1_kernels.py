import jax
import jax.numpy as jnp

from functools import partial

from .special_functions import spherical_harmonics, scaled_spherical_bessel_i
from .dataset_processing import get_cartesian_vectors


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def sigma(r, C_s, lambda_s):
    return C_s*jnp.exp(r/lambda_s)


def get_equal_elements_labels(a, b):
    # gets all pairs of indices in a, b which give equal a[i] = b[j]
    a_repeat = jnp.repeat(a, b.shape[0], axis=0)
    b_tile = jnp.tile(b, (a.shape[0], 1))
    where = jnp.nonzero(jnp.all(a_repeat==b_tile, axis=-1))[0]
    return jnp.stack([where//b.shape[0], where%b.shape[0]], axis=-1)


@partial(jax.jit, static_argnames="l_max")
def compute_wk_nu1_iijj(vectors1, vectors2, equal_element_labels, l_max, C_s, lambda_s):

    sh1 = spherical_harmonics(vectors1, l_max)
    r1 = jnp.sqrt(jnp.sum(vectors1**2, axis=1))

    sh2 = spherical_harmonics(vectors2, l_max)
    r2 = jnp.sqrt(jnp.sum(vectors2**2, axis=1))

    sh1_pairs = sh1[equal_element_labels[:, 0]]
    sh2_pairs = sh2[equal_element_labels[:, 1]]
    r1_pairs = r1[equal_element_labels[:, 0]]
    r2_pairs = r2[equal_element_labels[:, 1]]

    A = 1.0/(sigma(r1_pairs, C_s, lambda_s)**2+sigma(r2_pairs, C_s, lambda_s)**2)
    r_diff_pairs = r2_pairs - r1_pairs
    prefactors = 32*(jnp.pi)**3*(A/(2.0*jnp.pi))**(3/2)*jnp.exp(-A*r_diff_pairs**2/2.0)
    sbessi = scaled_spherical_bessel_i(A*r1_pairs*r2_pairs, l_max)

    wk_nu1_iijj = {}
    for l in range(l_max+1):
        sh1_pairs_l = sh1_pairs[:, l**2:(l+1)**2]
        sh2_pairs_l = sh2_pairs[:, l**2:(l+1)**2]
        sbessi_l = sbessi[:, l]
        wk_nu1_iijj[l] = ((prefactors/(2*l+1))*((-1)**l)*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]
        # wk_nu1_iijj[l] = (prefactors*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]
        # wk_nu1_iijj[l] = ((prefactors/(2*l+1))*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]

    return wk_nu1_iijj


def compute_wk_nu1(jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s):

    vectors1, labels1 = get_cartesian_vectors(jax_structures1, r_cut)
    vectors2, labels2 = get_cartesian_vectors(jax_structures2, r_cut)
    
    # Get all pairs for which the center and neighbor elements are the same:
    equal_element_labels = get_equal_elements_labels(labels1[:, [3, 4]], labels2[:, [3, 4]])

    wk_iijj = compute_wk_nu1_iijj(vectors1, vectors2, equal_element_labels, l_max, C_s, lambda_s)

    Sijai1_pairs = labels1[:, :4][equal_element_labels[:, 0]]
    Sijai2_pairs = labels2[:, :4][equal_element_labels[:, 1]]
    # assert jnp.all(Sijai1_pairs[:, 4] == Sijai2_pairs[:, 4])

    wk_nu1_ii = {}
    s1 = {}
    s2 = {}

    for a_i in all_species:

        ai_indices_pairs = jnp.nonzero(Sijai1_pairs[:, 4]==a_i)[0]
        Si1_ai_pairs = Sijai1_pairs[:, :2][ai_indices_pairs]
        Si2_ai_pairs = Sijai2_pairs[:, :2][ai_indices_pairs]

        unique_si1_indices, si1_unique_to_metadata, si1_metadata_to_unique = jnp.unique(Si1_ai_pairs, axis=0, return_index=True, return_inverse=True)
        unique_si2_indices, si2_unique_to_metadata, si2_metadata_to_unique = jnp.unique(Si2_ai_pairs, axis=0, return_index=True, return_inverse=True)
        n_si1 = len(unique_si1_indices)
        n_si2 = len(unique_si2_indices)
        indices_iijj_to_ii = n_si2*si1_metadata_to_unique+si2_metadata_to_unique

        wk_nu1_ii[a_i] = {}
        for l in range(l_max+1):
            wk_nu1_ii[a_i][(l, 1)] = jax.lax.scatter_add(
                jnp.zeros((n_si1*n_si2, 2*l+1, 2*l+1)),
                jnp.expand_dims(indices_iijj_to_ii, -1),
                wk_iijj[l][ai_indices_pairs],
                jax.lax.ScatterDimensionNumbers(update_window_dims=(1, 2), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
            ).reshape((n_si1, n_si2, 2*l+1, 2*l+1))

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
        s1[a_i] = Si1_ai_pairs[:, 0][si1_unique_to_metadata]
        s2[a_i] = Si2_ai_pairs[:, 0][si2_unique_to_metadata]

    return wk_nu1_ii, s1, s2
