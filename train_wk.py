import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ase
from ase import io
from math import factorial

from utils.dataset_processing import create_jax_structures
from utils.nu0_kernels import compute_wk_nu0
from utils.nu1_kernels import compute_wk_nu1
from utils.structure_wise import compute_stucture_wise_kernels
from utils.clebsch_gordan import get_cg_coefficients
from utils.wigner_iterations import compute_wks_high_order
from utils.error_measures import get_mae, get_rmse

import tqdm


np.random.seed(0)
n_train = 100
n_validation = 100
n_test = 100
batch_size = 20
train_validation_structures = ase.io.read("datasets/gold.xyz", ":200")
test_structures = ase.io.read("datasets/gold.xyz", "200:300")
# structures = [ase.Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
np.random.shuffle(train_validation_structures)
train_structures = train_validation_structures[:n_train]
validation_structures = train_validation_structures[n_train:]

train_targets = jnp.array([train_structure.info["elec. Free Energy [eV]"] for train_structure in train_structures])
validation_targets = jnp.array([validation_structure.info["elec. Free Energy [eV]"] for validation_structure in validation_structures])
test_targets = jnp.array([test_structure.info["elec. Free Energy [eV]"] for test_structure in test_structures])

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
all_species = [int(atomic_number) for atomic_number in all_species_jax]
print("All species:", all_species)
nu_max = 4
l_max = 3
cgs = get_cg_coefficients(l_max)
r_cut = 15.0
C_s = 0.1
lambda_s = 1.0



def compute_wks_single_batch(positions1, positions2, jax_structures1, jax_structures2):

    wks_nu0, s1_0, s2_0 = compute_wk_nu0(jax_structures1, jax_structures2, all_species)
    wks_nu1, s1, s2 = compute_wk_nu1(positions1, positions2, jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s)

    for a_i in all_species:
        assert jnp.all(s1[a_i]==s1_0[a_i])
        assert jnp.all(s2[a_i]==s2_0[a_i])

    invariant_wks = compute_wks_high_order(wks_nu1, all_species, nu_max, l_max, cgs)  # computes from nu = 1 to nu_max
    invariant_wks = [wks_nu0] + invariant_wks  # prepend nu = 0

    invariant_wks_per_structure = []
    for nu in range(nu_max+1):
        invariant_wks_per_structure.append(
            compute_stucture_wise_kernels(invariant_wks[nu], s1, s2)
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

    jax_batches1 = [create_jax_structures(batch, r_cut) for batch in batches1]
    jax_batches2 = [create_jax_structures(batch, r_cut) for batch in batches2]

    wks = jnp.empty((ntot1, ntot2, nu_max+1))
    idx1 = 0
    for jax_batch1 in tqdm.tqdm(jax_batches1):
        idx2 = 0
        for jax_batch2 in jax_batches2:

            wks_single_batch = compute_wks_single_batch(jax_batch1["positions"], jax_batch2["positions"], jax_batch1, jax_batch2)
            wks = wks.at[idx1:idx1+jax_batch1["n_structures"], idx2:idx2+jax_batch2["n_structures"], :].set(wks_single_batch)
            # print(jax.jacfwd(jax.jacrev(compute_wks_single_batch, argnums=0), argnums=1)(jax_batch1["positions"], jax_batch2["positions"], jax_batch1, jax_batch2).shape)

            idx2 += jax_batch2["n_structures"]
        idx1 += jax_batch1["n_structures"]
    return wks


train_train_kernels = compute_wks(train_structures, train_structures, batch_size)
validation_train_kernels = compute_wks(validation_structures, train_structures, batch_size)
test_train_kernels = compute_wks(test_structures, train_structures, batch_size)

# 3D grid search
optimization_target = "RMSE"
validation_best = 1e30
test_best = 1e30
print((f"The optimization target is {optimization_target}"))
for log_C0 in range(5, 15):
    for log_C in range(0, 10):
        for alpha in np.linspace(0, 2, 11):
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
            else:
                raise ValueError("The optimization target must be rmse or mae")
            print()
            print(log_C0, log_C, alpha)
            print(train_error, validation_error, test_error)
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
