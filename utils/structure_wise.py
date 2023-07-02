import jax
import jax.numpy as jnp


def compute_stucture_wise_kernels(atom_wise_kernels, s1, s2):

    # Get the number of structures in both kernel dimensions:
    n1 = jnp.max(jnp.concatenate([s1_ai for s1_ai in s1.values()]))
    n2 = jnp.max(jnp.concatenate([s2_ai for s2_ai in s2.values()]))
    structure_wise_kernels = jnp.zeros((n1, n2))

    for a_i, atom_wise_kernels_ai in atom_wise_kernels.items():
        s1_ai = s1[a_i]
        s2_ai = s2[a_i]
        scatter_indices = jnp.stack((jnp.repeat(s1_ai[:, jnp.newaxis], s2_ai.shape[0], axis=1), jnp.repeat(s2_ai[jnp.newaxis, :], s1_ai.shape[0], axis=0)), axis=-1)
        structure_wise_kernels = jax.lax.scatter_add(
            structure_wise_kernels,
            scatter_indices,
            atom_wise_kernels_ai,
            jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0, 1), scatter_dims_to_operand_dims=(0, 1))
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
