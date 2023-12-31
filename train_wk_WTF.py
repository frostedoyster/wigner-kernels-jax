import numpy as np
import jax
# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
import ase
from ase import io
from math import factorial
from functools import partial

from utils.dataset_processing import create_jax_structures
from utils.nu0_kernels import compute_wk_nu0
from utils.nu1_kernels import compute_wk_nu1
from utils.structure_wise import compute_stucture_wise_kernels
from utils.clebsch_gordan import get_cg_coefficients
from utils.wigner_iterations import compute_wks_high_order
from utils.error_measures import get_mae, get_rmse

import tqdm


np.random.seed(0)
n_train = 5
n_validation = 50
n_test = 1000
batch_size = 25

for C_s in np.geomspace(0.001, 1.0, 21):
    for lambda_s in np.geomspace(0.1, 10.0, 21):

        train_validation_structures = ase.io.read("datasets/ethanol1.extxyz", ":55")
        test_structures = ase.io.read("datasets/ethanol1.extxyz", "1000:2000")
        np.random.shuffle(train_validation_structures)
        train_structures = train_validation_structures[:n_train]
        validation_structures = train_validation_structures[n_train:]
        train_targets = jnp.array([train_structure.info["energy"] for train_structure in train_structures])
        validation_targets = jnp.array([validation_structure.info["energy"] for validation_structure in validation_structures])
        test_targets = jnp.array([test_structure.info["energy"] for test_structure in test_structures])


        def split_list(lst, n):
            """Yields successive n-sized chunks from lst."""
            list_of_chunks = []
            for i in range(0, len(lst), n):
                list_of_chunks.append(lst[i:i+n])
            return list_of_chunks


        all_species_jax = jnp.sort(jnp.unique(jnp.concatenate(
                [train_structure.numbers for train_structure in train_structures] + 
                [validation_structure.numbers for validation_structure in validation_structures] + 
                [test_structure.numbers for test_structure in test_structures]
            )))
        all_species = tuple([int(atomic_number) for atomic_number in all_species_jax])
        print("All species:", all_species)
        nu_max = 4
        l_max = 3
        cgs = get_cg_coefficients(l_max)
        r_cut = 10.0
        #C_s = 0.06
        #lambda_s = 1.4


        @partial(jax.jit, static_argnames=["n1", "n2"])
        def compute_wks_single_batch(positions1, positions2, jax_structures1, jax_structures2, n1, n2):

            wks_nu0, s1_0, s2_0 = compute_wk_nu0(jax_structures1, jax_structures2, all_species)
            wks_nu1, s1, s2 = compute_wk_nu1(positions1, positions2, jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s)

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


        def compute_wks(structures1, structures2, batch_size):
            batches1 = split_list(structures1, batch_size)
            batches2 = split_list(structures2, batch_size)
            n1 = [len(batch) for batch in batches1]
            n2 = [len(batch) for batch in batches2]
            ntot1 = sum(n1)
            ntot2 = sum(n2)

            jax_batches1 = [create_jax_structures(batch, all_species, r_cut) for batch in batches1]
            jax_batches2 = [create_jax_structures(batch, all_species, r_cut) for batch in batches2]
            """jax_batches1 = []
            for batch in tqdm.tqdm(batches1):
                jax_batches1.append(create_jax_structures(batch, r_cut))

            jax_batches2 = []
            for batch in tqdm.tqdm(batches2):
                jax_batches2.append(create_jax_structures(batch, r_cut))"""
                

            wks = jnp.empty((ntot1, ntot2, nu_max+1))
            idx1 = 0
            for jax_batch1 in tqdm.tqdm(jax_batches1):
                idx2 = 0
                for jax_batch2 in jax_batches2:

                    wks_single_batch = compute_wks_single_batch(jax_batch1["positions"], jax_batch2["positions"], jax_batch1, jax_batch2, jax_batch1["n_structures"], jax_batch2["n_structures"])
                    wks = wks.at[idx1:idx1+jax_batch1["n_structures"], idx2:idx2+jax_batch2["n_structures"], :].set(wks_single_batch)
                    # print(jax.jacfwd(jax.jacrev(compute_wks_single_batch, argnums=0), argnums=1)(jax_batch1["positions"], jax_batch2["positions"], jax_batch1, jax_batch2, jax_batch1["n_structures"], jax_batch2["n_structures"]).shape)

                    idx2 += jax_batch2["n_structures"]
                idx1 += jax_batch1["n_structures"]
            return wks


        train_train_kernels = compute_wks(train_structures, train_structures, batch_size)
        for nu in range(nu_max+1):
            print(train_train_kernels[:, :, nu])
            print()
        exit()

        validation_train_kernels = compute_wks(validation_structures, train_structures, batch_size)
        test_train_kernels = compute_wks(test_structures, train_structures, batch_size)

        # 3D grid search
        optimization_target = "MAE"
        validation_best = 1e30
        test_best = 1e30
        print((f"The optimization target is {optimization_target}"))
        for log_C0 in tqdm.tqdm(range(5, 15)):
            for log_C in range(-5, 5):
                for alpha in np.linspace(0, 5, 101):
                    C0 = 10**log_C0
                    C = 10**log_C
                    nu_coefficient_vector = jnp.array([C0] + [C*(alpha**nu)/factorial(nu) for nu in range(1, nu_max+1)])
                    train_train_kernel = train_train_kernels @ nu_coefficient_vector
                    validation_train_kernel = validation_train_kernels @ nu_coefficient_vector
                    test_train_kernel = test_train_kernels @ nu_coefficient_vector
                    c = jnp.linalg.solve(train_train_kernel+jnp.eye(n_train), train_targets)
                    if optimization_target == "RMSE":
                        train_error = get_rmse(train_targets, train_train_kernel @ c)
                        validation_error = get_rmse(validation_targets, validation_train_kernel @ c)
                        test_error = get_rmse(test_targets, test_train_kernel @ c)
                    elif optimization_target == "MAE":
                        train_error = get_mae(train_targets, train_train_kernel @ c)
                        validation_error = get_mae(validation_targets, validation_train_kernel @ c)
                        test_error = get_mae(test_targets, test_train_kernel @ c)
                    else:
                        raise ValueError("The optimization target must be rmse or mae")
                    # print()
                    # print(log_C0, log_C, alpha)
                    # print(train_error, validation_error, test_error)
                    if validation_error < validation_best:
                        validation_best = validation_error
                        test_best = test_error
                        log_C0_best = log_C0
                        log_C_best = log_C
                        alpha_best = alpha

        print(f"Best optimization parameters: log_C0={log_C0_best}, log_C={log_C_best}, alpha={alpha_best}")
        print(f"Best validation error: {validation_best}")

        print()
        print(f"Test error ({optimization_target}):")
        print(test_best)
