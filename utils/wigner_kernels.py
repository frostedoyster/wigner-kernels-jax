import jax
import jax.numpy as jnp
from functools import partial
import tqdm
from .dataset_processing import create_jax_structures
from .nu0_kernels import compute_wk_nu0
from .nu1_kernels import compute_wk_nu1
from .wigner_iterations import compute_wks_high_order
from .structure_wise import compute_stucture_wise_kernels


def split_list(lst, n):
    """Yields successive n-sized chunks from lst."""
    list_of_chunks = []
    for i in range(0, len(lst), n):
        list_of_chunks.append(lst[i:i+n])
    return list_of_chunks


@partial(jax.jit, static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"])
def compute_wks_single_batch(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s):

    wks_nu0, s1_0, s2_0 = compute_wk_nu0(jax_structures_1, jax_structures_2, all_species)
    wks_nu1, s1, s2 = compute_wk_nu1(positions_1, positions_2, jax_structures_1, jax_structures_2, all_species, l_max, C_s, lambda_s)

    """for a_i in all_species:
        assert jnp.all(s1[a_i] == s1_0[a_i])
        assert jnp.all(s2[a_i] == s2_0[a_i])"""

    invariant_wks = compute_wks_high_order(wks_nu1, all_species, nu_max, l_max, cgs)  # computes from nu = 1 to nu_max
    invariant_wks = [wks_nu0] + invariant_wks  # prepend nu = 0

    invariant_wks_per_structure = []
    for nu in range(nu_max+1):
        invariant_wks_per_structure.append(
            compute_stucture_wise_kernels(invariant_wks[nu], s1, s2, n1, n2)
        )

    invariant_wks_per_structure = jnp.stack(invariant_wks_per_structure, axis=-1)
    return invariant_wks_per_structure


@partial(jax.jit, static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"])
def compute_wks_single_batch_sum_1(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s):
    wks_single_batch = compute_wks_single_batch(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
    return jnp.sum(wks_single_batch, axis=0)

compute_wks_single_batch_jac_1 = jax.jit(
    jax.jacrev(compute_wks_single_batch_sum_1, argnums=0),
    static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"]
)

@partial(jax.jit, static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"])
def compute_wks_single_batch_sum_2(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s):
    wks_single_batch = compute_wks_single_batch(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
    return jnp.sum(wks_single_batch, axis=1)

compute_wks_single_batch_jac_2 = jax.jit(
    jax.jacrev(compute_wks_single_batch_sum_2, argnums=1),
    static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"]
)

@partial(jax.jit, static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"])
def compute_wks_single_batch_sum_1_2(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s):
    wks_single_batch = compute_wks_single_batch(positions_1, positions_2, jax_structures_1, jax_structures_2, n1, n2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
    return jnp.sum(wks_single_batch, axis=(0, 1))

compute_wks_single_batch_hess = jax.jit(
    jax.jacfwd(
        jax.jacrev(
            compute_wks_single_batch_sum_1_2,
            argnums=0
        ),
        argnums=1
    ),
    static_argnames=["n1", "n2", "all_species", "l_max", "nu_max", "C_s"]
)


def compute_wks(structures1, structures2, all_species, r_cut, l_max, nu_max, cgs, batch_size, C_s, lambda_s):
    batches1 = split_list(structures1, batch_size)
    batches2 = split_list(structures2, batch_size)
    ntot1 = sum([len(batch) for batch in batches1])
    ntot2 = sum([len(batch) for batch in batches2])

    jax_batches_1 = [create_jax_structures(batch, all_species, r_cut) for batch in batches1]
    jax_batches_2 = [create_jax_structures(batch, all_species, r_cut) for batch in batches2]

    wks = jnp.empty((ntot1, ntot2, nu_max+1))
    idx1 = 0
    for jax_batch_1 in tqdm.tqdm(jax_batches_1):
        idx2 = 0
        for jax_batch_2 in jax_batches_2:

            wks_single_batch = compute_wks_single_batch(jax_batch_1["positions"], jax_batch_2["positions"], jax_batch_1, jax_batch_2, jax_batch_1["n_structures"], jax_batch_2["n_structures"], all_species, l_max, nu_max, cgs, C_s, lambda_s)
            wks = wks.at[idx1:idx1+jax_batch_1["n_structures"], idx2:idx2+jax_batch_2["n_structures"], :].set(wks_single_batch)
            # print(jax.jacfwd(jax.jacrev(compute_wks_single_batch, argnums=0), argnums=1)(jax_batch_1["positions"], jax_batch_2["positions"], jax_batch_1, jax_batch_2, jax_batch_1["n_structures"], jax_batch_2["n_structures"]).shape)

            idx2 += jax_batch_2["n_structures"]
        idx1 += jax_batch_1["n_structures"]
    return wks


def compute_wks_with_derivatives(structures1, structures2, all_species, r_cut, l_max, nu_max, cgs, batch_size, C_s, lambda_s):
    batches1 = split_list(structures1, batch_size)
    batches2 = split_list(structures2, batch_size)
    ntot1 = sum([len(batch) for batch in batches1])
    ntot2 = sum([len(batch) for batch in batches2])

    jax_batches_1 = [create_jax_structures(batch, all_species, r_cut) for batch in batches1]
    jax_batches_2 = [create_jax_structures(batch, all_species, r_cut) for batch in batches2]
    ntot1_der = 3*sum([jax_batch["positions"].shape[0] for jax_batch in jax_batches_1])
    ntot2_der = 3*sum([jax_batch["positions"].shape[0] for jax_batch in jax_batches_2])

    wks = jnp.empty((ntot1+ntot1_der, ntot2+ntot2_der, nu_max+1))
    idx1 = 0
    idx1_der = ntot1
    for jax_batch_1 in tqdm.tqdm(jax_batches_1):
        positions_1 = jax_batch_1["positions"]
        n_structures_1 = jax_batch_1["n_structures"]
        n_atoms_1 = positions_1.shape[0]

        idx2 = 0
        idx2_der = ntot2
        for jax_batch_2 in jax_batches_2:
            positions_2 = jax_batch_2["positions"]
            n_structures_2 = jax_batch_2["n_structures"]
            n_atoms_2 = positions_2.shape[0]

            wks_single_batch = compute_wks_single_batch(positions_1, positions_2, jax_batch_1, jax_batch_2, n_structures_1, n_structures_2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
            wks = wks.at[idx1:idx1+n_structures_1, idx2:idx2+n_structures_2, :].set(wks_single_batch)
            wks_single_batch_jac_1 = compute_wks_single_batch_jac_1(positions_1, positions_2, jax_batch_1, jax_batch_2, n_structures_1, n_structures_2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
            wks = wks.at[idx1_der:idx1_der+3*n_atoms_1, idx2:idx2+n_structures_2, :].set(wks_single_batch_jac_1.reshape(n_structures_2, nu_max+1, 3*n_atoms_1).swapaxes(1, 2).swapaxes(0, 1))
            wks_single_batch_jac_2 = compute_wks_single_batch_jac_2(positions_1, positions_2, jax_batch_1, jax_batch_2, n_structures_1, n_structures_2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
            wks = wks.at[idx1:idx1+n_structures_1, idx2_der:idx2_der+3*n_atoms_2, :].set(wks_single_batch_jac_2.reshape(n_structures_1, nu_max+1, 3*n_atoms_2).swapaxes(1, 2))
            wks_single_batch_hess = compute_wks_single_batch_hess(positions_1, positions_2, jax_batch_1, jax_batch_2, n_structures_1, n_structures_2, all_species, l_max, nu_max, cgs, C_s, lambda_s)
            wks = wks.at[idx1_der:idx1_der+3*n_atoms_1, idx2_der:idx2_der+3*n_atoms_2, :].set(wks_single_batch_hess.reshape(nu_max+1, 9*n_atoms_1*n_atoms_2).swapaxes(0, 1).reshape(3*n_atoms_1, 3*n_atoms_2, nu_max+1))

            idx2 += n_structures_2
            idx2_der += 3*n_atoms_2
        idx1 += n_structures_1
        idx1_der += 3*n_atoms_1
    return wks


"""def compute_wks_symm(structures, all_species, r_cut, l_max, nu_max, cgs, batch_size, C_s, lambda_s):
    batches = split_list(structures, batch_size)
    ntot = sum([len(batch) for batch in batches])
    jax_batches = [create_jax_structures(batch, all_species, r_cut) for batch in batches]

    wks = jnp.empty((ntot, ntot, nu_max+1))
    idx1 = 0
    for idx_batch_1 in tqdm.tqdm(range(len(jax_batches))):
        idx2 = 0
        for idx_batch_2 in range(len(jax_batches)):

            if idx_batch_1 == idx_batch_2:
                jax_batch = jax_batches[idx_batch_1]
                positions = jax_batch["positions"]
                n_structures = jax_batch["n_structures"]
                n_atoms = positions.shape[0]
                wks_single_batch = compute_wks_symm_single_batch(positions, jax_batch, n_structures, all_species, l_max, nu_max, cgs, C_s, lambda_s)
                wks = wks.at[idx1:idx1+n_structures, idx2:idx2+n_structures, :].set(wks_single_batch)
            else:
                jax_batch_1 = jax_batches[idx_batch_1]
                jax_batch_2 = jax_batches[idx_batch_2]
                wks_single_batch = compute_wks_single_batch(jax_batch_1["positions"], jax_batch_2["positions"], jax_batch_1, jax_batch_2, jax_batch_1["n_structures"], jax_batch_2["n_structures"], all_species, l_max, nu_max, cgs, C_s, lambda_s)
                wks = wks.at[idx1:idx1+jax_batch_1["n_structures"], idx2:idx2+jax_batch_2["n_structures"], :].set(wks_single_batch)


            idx2 += jax_batches[idx_batch_2]["n_structures"]
        idx1 += jax_batches[idx_batch_1]["n_structures"]
    
    return wks"""


"""@partial(jax.jit, static_argnames=["n", "all_species", "l_max", "nu_max", "C_s", "lambda_s"])
def compute_wks_symm_single_batch_jac(positions, jax_structures, n, all_species, l_max, nu_max, cgs, C_s, lambda_s):
    compute_wks_symm_single_batch_sum():
    jac = jax.jacrev()"""


"""def compute_wks_symm_with_derivatives(structures, batch_size):
    batches = split_list(structures, batch_size)
    ntot = sum([len(batch) for batch in batches])
    jax_batches = [create_jax_structures(batch, all_species, r_cut) for batch in batches]
    ntot_der = 3*sum([jax_batch["positions"].shape[0] for jax_batch in jax_batches])

    wks = jnp.empty((ntot+ntot_der, ntot+ntot_der, nu_max+1))
    idx1 = 0
    idx1_der = ntot
    for idx_batch_1 in tqdm.tqdm(range(len(jax_batches))):
        idx2 = 0
        idx2_der = ntot
        for idx_batch_2 in range(len(jax_batches)):

            if idx_batch_1 == idx_batch_2:
                jax_batch = jax_batches[idx_batch_1]
                positions = jax_batch["positions"]
                n_structures = jax_batch["n_structures"]
                n_atoms = positions.shape[0]
                wks_single_batch = compute_wks_single_batch_symm(positions, jax_batch, n_structures)
                wks = wks.at[idx1:idx1+n_structures, idx2:idx2+n_structures, :].set(wks_single_batch)
                wks_single_batch_jac = compute_wks_single_batch_symm_jac(positions, jax_batch, n_structures)
                # wks = wks.at[idx1:idx1+n_structures, idx2:idx2_der+3*n_atoms, :].set(wks_single_batch_jac.reshape(nu_max+1, 3*n_atoms).T)
                #wks = wks.at[idx1:idx1+n_structures, idx2:idx2_der+3*n_atoms, :].set(wks_single_batch_jac.reshape(nu_max+1, 3*n_atoms).T)
                #wks_single_batch_hess = compute_wks_single_batch_symm_hess(positions, jax_batch, n_structures)
                #wks = wks.at[idx1:idx1+n_structures, idx2:idx2+n_structures, :].set(wks_single_batch)

            else:
                jax_batch_1 = jax_batches[idx_batch_1]
                jax_batch_2 = jax_batches[idx_batch_2]
                wks_single_batch = compute_wks_single_batch(jax_batch_1["positions"], jax_batch_2["positions"], jax_batch_1, jax_batch_2, jax_batch_1["n_structures"], jax_batch_2["n_structures"])
                wks = wks.at[idx1:idx1+jax_batch_1["n_structures"], idx2:idx2+jax_batch_2["n_structures"], :].set(wks_single_batch)
                # print(jax.jacfwd(jax.jacrev(compute_wks_single_batch, argnums=0), argnums=1)(jax_batch_1["positions"], jax_batch_2["positions"], jax_batch_1, jax_batch_2, jax_batch_1["n_structures"], jax_batch_2["n_structures"]).shape)

            idx2 += jax_batches[idx_batch_2]["n_structures"]
            idx2_der += 3*jax_batches[idx_batch_2]["positions"].shape[0]
        idx1 += jax_batches[idx_batch_1]["n_structures"]
        idx2_der += 3*jax_batches[idx_batch_1]["positions"].shape[0]
    
    return wks"""


"""@partial(jax.jit, static_argnames=["n", "all_species", "l_max", "nu_max", "C_s", "lambda_s"])
def compute_wks_symm_single_batch(positions, jax_structures, n, all_species, l_max, nu_max, cgs, C_s, lambda_s):
    return compute_wks_single_batch(positions, positions, jax_structures, jax_structures, n, n, all_species, l_max, nu_max, cgs, C_s, lambda_s)"""

"""compute_wks_single_batch_symm_jac = jax.jit(jax.jacrev(compute_wks_single_batch_symm_sum, argnums=0), static_argnames=["n"])
compute_wks_single_batch_symm_hess = jax.jit(jax.jacfwd(compute_wks_single_batch_symm_jac, argnums=0), static_argnames=["n"])"""
