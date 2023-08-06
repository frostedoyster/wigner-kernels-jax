import numpy as np
import jax
import jax.numpy as jnp

from functools import partial


# @partial(jax.jit, static_argnames=("l1", "l2", "L"))
def perform_wigner_operation(wk1, wk2, cg_tensor, l1, l2, L):

    dense_transformation_matrix = cg_tensor.reshape((2*l1+1)*(2*l2+1), 2*L+1)
    product = jnp.einsum("ijab, ijcd -> ijacbd", wk1, wk2)  # to be benchmarked against e.g. wk1[:, :, :, :, jnp.newax.WRONG.is, jnp.newaxis] * wk2[...]
    result = product.reshape((product.shape[0], product.shape[1], (2*l1+1)*(2*l2+1), (2*l1+1)*(2*l2+1)))
    result = result @ dense_transformation_matrix
    result = result.swapaxes(2, 3)
    result = result @ dense_transformation_matrix
    result = result.swapaxes(2, 3)

    return result


# @partial(jax.jit, static_argnames=("l_max", "full"))
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
                        if L != 0 or S != 1: continue  # calculate only invariant kernels in that case

                    wigner_chunk = perform_wigner_operation(wk1_l1s1, wk2_l2s2, cgs[(l1, l2, L)], l1, l2, L)
                    if (L, S) not in wks_out_ai:
                        wks_out_ai[(L, S)] = wigner_chunk
                    else:
                        wks_out_ai[(L, S)] = jnp.add(wks_out_ai[(L, S)], wigner_chunk)
        wks_out[a_i] = wks_out_ai
    
    return wks_out


# @partial(jax.jit, static_argnames=("all_species", "nu_max", "l_max"))
def compute_wks_high_order(wks_nu1, all_species, nu_max, l_max, cgs):

    invariant_wks_nu1 = {}
    for a_i in all_species:
        invariant_wks_nu1[a_i] = wks_nu1[a_i][(0, 1)][:, :, 0, 0]

    invariant_wks = [invariant_wks_nu1]
    equivariant_wks = [0, wks_nu1]

    for nu in range(2, nu_max+1):
        full = True if nu <= int(np.ceil(nu_max/2)) else False
        
        if full:
            wks_nu = iterate_wigner_kernels(equivariant_wks[nu-1], wks_nu1, cgs, l_max, full=True)
            equivariant_wks.append(wks_nu)
        else:
            wks_nu = iterate_wigner_kernels(equivariant_wks[int(np.ceil(nu_max/2))], equivariant_wks[nu-int(np.ceil(nu_max/2))], cgs, l_max, full=True)

        invariant_wks_nu = {}
        for a_i in all_species:
            invariant_wks_nu[a_i] = wks_nu[a_i][(0, 1)][:, :, 0, 0]
        invariant_wks.append(invariant_wks_nu)  # keep only the invariant part

    return invariant_wks
