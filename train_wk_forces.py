import numpy as np
import jax
# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
import ase
from ase import io
from math import factorial
from functools import partial

from utils.clebsch_gordan import get_cg_coefficients
from utils.error_measures import get_mae, get_rmse
from utils.wigner_kernels import compute_wks_with_derivatives

import tqdm


np.random.seed(0)
n_train = 45
n_validation = 5
n_test = 50
batch_size = 1

train_validation_structures = ase.io.read("datasets/ethanol1.extxyz", "0:1000")
test_structures = ase.io.read("datasets/ethanol1.extxyz", "1000:2000")
np.random.shuffle(train_validation_structures)
np.random.shuffle(test_structures)
train_structures = train_validation_structures[:n_train]
validation_structures = train_validation_structures[n_train:n_train+n_validation]
test_structures = test_structures[:n_test]

train_energies = jnp.array([train_structure.info["energy"] for train_structure in train_structures])
validation_energies = jnp.array([validation_structure.info["energy"] for validation_structure in validation_structures])
test_energies = jnp.array([test_structure.info["energy"] for test_structure in test_structures])

train_forces = jnp.concatenate([jnp.array(train_structure.get_forces()) for train_structure in train_structures])
validation_forces = jnp.concatenate([jnp.array(validation_structure.get_forces()) for validation_structure in validation_structures])
test_forces = jnp.concatenate([jnp.array(test_structure.get_forces()) for test_structure in test_structures])

train_targets = jnp.concatenate([train_energies, -train_forces.flatten()])
validation_targets = jnp.concatenate([validation_energies, -validation_forces.flatten()])
test_targets = jnp.concatenate([test_energies, -test_forces.flatten()])

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
C_s = 0.1
lambda_s = jnp.array([0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.66])

train_train_kernels = compute_wks_with_derivatives(train_structures, train_structures, all_species, r_cut, l_max, nu_max, cgs, batch_size, C_s, lambda_s)
for nu in range(nu_max+1):
    print(train_train_kernels[:, :, nu])
    print()

validation_train_kernels = compute_wks_with_derivatives(validation_structures, train_structures, all_species, r_cut, l_max, nu_max, cgs, batch_size, C_s, lambda_s)
test_train_kernels = compute_wks_with_derivatives(test_structures, train_structures, all_species, r_cut, l_max, nu_max, cgs, batch_size, C_s, lambda_s)

# 3D grid search
optimization_target = "MAE"
validation_score_best = 1e30
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
            c = jnp.linalg.solve(train_train_kernel+jnp.eye(train_train_kernel.shape[0]), train_targets)
            if optimization_target == "RMSE":
                train_error_forces = get_rmse(train_targets[n_train:], (train_train_kernel @ c)[n_train:])
                validation_error_forces = get_rmse(validation_targets[n_validation:], (validation_train_kernel @ c)[n_validation:])
                test_error_forces = get_rmse(test_targets[n_test:], (test_train_kernel @ c)[n_test:])
                train_error_energies = get_rmse(train_targets[:n_train], (train_train_kernel @ c)[:n_train])
                validation_error_energies = get_rmse(validation_targets[:n_validation], (validation_train_kernel @ c)[:n_validation])
                test_error_energies = get_rmse(test_targets[:n_test], (test_train_kernel @ c)[:n_test])
            elif optimization_target == "MAE":
                train_error_forces = get_mae(train_targets[n_train:], (train_train_kernel @ c)[n_train:])
                validation_error_forces = get_mae(validation_targets[n_validation:], (validation_train_kernel @ c)[n_validation:])
                test_error_forces = get_mae(test_targets[n_test:], (test_train_kernel @ c)[n_test:])
                train_error_energies = get_mae(train_targets[:n_train], (train_train_kernel @ c)[:n_train])
                validation_error_energies = get_mae(validation_targets[:n_validation], (validation_train_kernel @ c)[:n_validation])
                test_error_energies = get_mae(test_targets[:n_test], (test_train_kernel @ c)[:n_test])
            else:
                raise ValueError("The optimization target must be rmse or mae")
            print()
            print(log_C0, log_C, alpha)
            print(train_error_energies, validation_error_energies, test_error_energies)
            validation_score = validation_error_energies+validation_error_forces
            if validation_score < validation_score_best:
                validation_score_best = validation_score
                validation_best_forces = validation_error_forces
                validation_best_energies = validation_error_energies
                test_best_forces = test_error_forces
                test_best_energies = test_error_energies
                log_C0_best = log_C0
                log_C_best = log_C
                alpha_best = alpha

print(f"Best optimization parameters: log_C0={log_C0_best}, log_C={log_C_best}, alpha={alpha_best}")
print(f"Best validation error: {validation_best_energies} {validation_best_forces}")

print()
print(f"Test error ({optimization_target}):")
print(test_best_energies, test_best_forces)
