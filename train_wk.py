import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ase
from ase import io
from utils.dataset_processing import JaxStructures, get_cartesian_vectors
from utils.special_functions import spherical_harmonics, scaled_spherical_bessel_i

structures = ase.io.read("datasets/gold.xyz", ":10")
# structures = [ase.Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
jax_structures = JaxStructures(structures)
jax_structures1 = jax_structures
jax_structures2 = jax_structures

all_species_jax = jnp.sort(jnp.unique(jnp.concatenate([train_structure.numbers for train_structure in structures] + [test_structure.numbers for test_structure in structures])))
all_species = [int(atomic_number) for atomic_number in all_species_jax]
print(all_species)
l_max = 3

@jax.jit
@jax.vmap
def sigma(r):
    return 0.1*jnp.exp(1.0*r)


def get_equal_elements_labels(a, b):
    # gets all pairs of indices in a, b which give equal a[i] = b[j]
    a_repeat = jnp.repeat(a, b.shape[0], axis=0)
    b_tile = jnp.tile(b, (a.shape[0], 1))
    where = jnp.nonzero(jnp.all(a_repeat==b_tile, axis=-1))[0]
    return jnp.stack([where//b.shape[0], where%b.shape[0]], axis=-1)


def compute_wk_nu1_iijj(vectors1, vectors2, labels1, labels2, l_max):

    #print(vectors1)
    #print(vectors2)

    sh1 = spherical_harmonics(vectors1, l_max)
    r1 = jnp.sqrt(jnp.sum(vectors1**2, axis=1))

    sh2 = spherical_harmonics(vectors2, l_max)
    r2 = jnp.sqrt(jnp.sum(vectors2**2, axis=1))

    # print(r1, r2)

    # Get all pairs for which the center and neighbor elements are the same:
    equal_element_labels = get_equal_elements_labels(labels1[:, [3, 4]], labels2[:, [3, 4]])

    sh1_pairs = sh1[equal_element_labels[:, 0]]
    sh2_pairs = sh2[equal_element_labels[:, 1]]
    r1_pairs = r1[equal_element_labels[:, 0]]
    r2_pairs = r2[equal_element_labels[:, 1]]

    A = 1.0/(sigma(r1_pairs)**2+sigma(r2_pairs)**2)
    r_diff_pairs = r2_pairs - r1_pairs
    prefactors = 32*(jnp.pi)**3*(A/(2.0*jnp.pi))**(3/2)*jnp.exp(-A*r_diff_pairs**2/2.0)
    sbessi = scaled_spherical_bessel_i(A*r1_pairs*r2_pairs, l_max)

    wk_nu1_iijj = {}
    for l in range(l_max+1):
        sh1_pairs_l = sh1_pairs[:, l**2:(l+1)**2]
        sh2_pairs_l = sh2_pairs[:, l**2:(l+1)**2]
        sbessi_l = sbessi[:, l]
        # wk_nu1_iijj[l] = ((prefactors/(2*l+1))*((-1)**l)*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]
        wk_nu1_iijj[l] = (prefactors*sbessi_l)[:, None, None] * sh1_pairs_l[:, :, None] * sh2_pairs_l[:, None, :]

    Sijai1_pairs = labels1[:, :4][equal_element_labels[:, 0]]
    Sijai2_pairs = labels2[:, :4][equal_element_labels[:, 1]]
    # assert jnp.all(Sijai1_pairs[:, 4] == Sijai2_pairs[:, 4])

    return wk_nu1_iijj, Sijai1_pairs, Sijai2_pairs


def compute_wk_nu1(jax_structures1, jax_structures2, all_species, l_max):

    vectors1, labels1 = get_cartesian_vectors(jax_structures1, 15.0)
    vectors2, labels2 = get_cartesian_vectors(jax_structures2, 15.0)
    wk_iijj, Sijai1_pairs, Sijai2_pairs = compute_wk_nu1_iijj(vectors1, vectors2, labels1, labels2, l_max)

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



def compute_stucture_wise_kernels(atom_wise_kernels, s1, s2):

    # Get the number of structures in both kernel dimensions:
    n1 = jnp.max(jnp.concatenate([s1_ai for s1_ai in s1.values()]))
    n2 = jnp.max(jnp.concatenate([s2_ai for s2_ai in s2.values()]))

    structure_wise_kernels = {}
    for key in atom_wise_kernels[list(atom_wise_kernels.keys())[0]].keys():
        l = key[0]
        structure_wise_kernels[key] = jnp.zeros((n1, n2, 2*l+1, 2*l+1))

    for a_i, atom_wise_kernels_ai in atom_wise_kernels.items():
        s1_ai = s1[a_i]
        s2_ai = s2[a_i]
        scatter_indices = jnp.stack((jnp.repeat(s1_ai[:, jnp.newaxis], s2_ai.shape[0], axis=1), jnp.repeat(s2_ai[jnp.newaxis, :], s1_ai.shape[0], axis=0)), axis=-1)
        for key in atom_wise_kernels_ai.keys():
            structure_wise_kernels[key] = jax.lax.scatter_add(
                structure_wise_kernels[key],
                scatter_indices,
                atom_wise_kernels_ai[key],
                jax.lax.ScatterDimensionNumbers(update_window_dims=(2, 3), inserted_window_dims=(0, 1), scatter_dims_to_operand_dims=(0, 1))
            )

            """
            import jax
            import jax.numpy as jnp
            import numpy as np
            np.random.seed(0)

            A = jnp.array(np.random.rand(4, 4, 2, 2))
            indices1 = jnp.array([1, 0, 1, 1])
            indices2 = jnp.array([1, 0, 0, 0])
            indices = jnp.stack((jnp.repeat(indices1[:, jnp.newaxis], indices2.shape[0], axis=1), jnp.repeat(indices2[jnp.newaxis, :], indices1.shape[0], axis=0)), axis=-1)
            print(indices[2, 3])
            print(indices.shape)
            B = jax.lax.scatter_add(
                jnp.zeros((2, 2, 2, 2)),
                indices, 
                A,
                jax.lax.ScatterDimensionNumbers(update_window_dims=(2, 3), inserted_window_dims=(0, 1), scatter_dims_to_operand_dims=(0, 1))
            )
            print()
            print(B[0, 0]-A[1, 1]-A[1, 2]-A[1, 3])
            print(B[1, 0]-A[0, 1]-A[0, 2]-A[0, 3]-A[2, 1]-A[2, 2]-A[2, 3]-A[3, 1]-A[3, 2]-A[3, 3])
            print(B[0, 1]-A[1, 0])
            print(B[1, 1]-A[0, 0]-A[2, 0]-A[3, 0])
            """

    return structure_wise_kernels

wks, s1, s2 = compute_wk_nu1(jax_structures1, jax_structures2, all_species, l_max)
summed_wks = compute_stucture_wise_kernels(wks, s1, s2)

print(real/summed_wks[(0, 1)][:6, :6, 0, 0])
