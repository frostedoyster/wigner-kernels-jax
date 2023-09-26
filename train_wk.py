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
from utils.wigner_kernels import compute_wks
from utils.spliner import get_LE_splines

import tqdm

np.random.seed(0)
n_train = 1000
n_validation = 1000
n_test = 1000
batch_size = 50

train_validation_structures = ase.io.read("datasets/rmd17/ethanol1.extxyz", ":")
# np.random.shuffle(train_validation_structures)
train_structures = train_validation_structures[:n_train]
validation_structures = train_validation_structures[n_train:n_train+n_validation]
test_structures = train_validation_structures[n_train:n_train+n_test]
train_targets = jnp.array([train_structure.info["energy"] for train_structure in train_structures])*43.3
validation_targets = jnp.array([validation_structure.info["energy"] for validation_structure in validation_structures])*43.3
test_targets = jnp.array([test_structure.info["energy"] for test_structure in test_structures])*43.3

all_species_jax = jnp.sort(jnp.unique(jnp.concatenate(
        [train_structure.numbers for train_structure in train_structures] + 
        [validation_structure.numbers for validation_structure in validation_structures] + 
        [test_structure.numbers for test_structure in test_structures]
    )))
all_species = tuple([int(atomic_number) for atomic_number in all_species_jax])
print("All species:", all_species)
nu_max = 4
l_max = 3
n_max = 35
cgs = get_cg_coefficients(l_max)
r_cut = 10.0
#C_s = jnp.array([0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.75, 0.71, 0.66, 0.57]) * 0.03
#lambda_s = jnp.array([0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.75, 0.71, 0.66, 0.57]) * 0.5
C_s = 0.2
lambda_s = 3.0

spline_positions, spline_values, spline_derivatives = get_LE_splines(l_max, n_max, r_cut, C_s, lambda_s, 1e-4)
radial_splines = {
    "positions": jnp.array(spline_positions),
    "values": jnp.array(spline_values),
    "derivatives": jnp.array(spline_derivatives)
}

train_train_kernels = compute_wks(train_structures, train_structures, all_species, r_cut, l_max, n_max, nu_max, cgs, batch_size, radial_splines)

print()
print("Printing a few representative kernels:")
print(train_structures[0].positions)
print(train_structures[1].positions)
for nu in range(nu_max+1):
    print(f"nu={nu}")
    print(train_train_kernels[:6, :6, nu])
    print()

validation_train_kernels = compute_wks(validation_structures, train_structures, all_species, r_cut, l_max, n_max, nu_max, cgs, batch_size, radial_splines)
test_train_kernels = compute_wks(test_structures, train_structures, all_species, r_cut, l_max, n_max, nu_max, cgs, batch_size, radial_splines)

# 3D grid search
optimization_target = "MAE"
validation_best = 1e90
test_best = 1e90
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
