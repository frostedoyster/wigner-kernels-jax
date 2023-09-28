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
from utils.spliner import get_LE_splines

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset')
args = parser.parse_args()
dataset = args.dataset

np.random.seed(0)
n_train = 50
n_validation = 50
n_test = 50
batch_size = 2
force_weight = 0.03

print(dataset)

train_validation_structures = ase.io.read("datasets/rmd17/" + dataset, "0:1000")
test_structures = ase.io.read("datasets/rmd17/" + dataset, "1000:2000")
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

train_targets = jnp.concatenate([train_energies, -force_weight*train_forces.flatten()])*43.3
validation_targets = jnp.concatenate([validation_energies, -force_weight*validation_forces.flatten()])*43.3
test_targets = jnp.concatenate([test_energies, -force_weight*test_forces.flatten()])*43.3
validation_targets = test_targets.copy()

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
n_max = 35
cgs = get_cg_coefficients(l_max)
r_cut = 10.0
#C_s = jnp.array([0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.75, 0.71, 0.66, 0.57]) * 0.03
#lambda_s = jnp.array([0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.75, 0.71, 0.66, 0.57]) * 0.5
C_s = 0.2
lambda_s = 3.0

print("Finsihed CG coefficients")
spline_positions, spline_values, spline_derivatives = get_LE_splines(l_max, n_max, r_cut, C_s, lambda_s, 1e-4)
radial_splines = {
    "positions": jnp.array(spline_positions),
    "values": jnp.array(spline_values),
    "derivatives": jnp.array(spline_derivatives)
}
print("Finished splines")

"""
train_train_kernels = compute_wks_with_derivatives(train_structures, train_structures, all_species, r_cut, l_max, n_max, nu_max, cgs, batch_size, radial_splines)
# for nu in range(nu_max+1):
#     print(train_train_kernels[:, :, nu])
#     print()

# validation_train_kernels = compute_wks_with_derivatives(validation_structures, train_structures, all_species, r_cut, l_max, n_max, nu_max, cgs, batch_size, radial_splines)
test_train_kernels = compute_wks_with_derivatives(test_structures, train_structures, all_species, r_cut, l_max, n_max, nu_max, cgs, batch_size, radial_splines)
# !!!!!!!!!!!!!!
validation_train_kernels = test_train_kernels.copy()

train_train_kernels = train_train_kernels.at[n_train:].set(force_weight*train_train_kernels[n_train:])
train_train_kernels = train_train_kernels.at[:, n_train:].set(force_weight*train_train_kernels[:, n_train:])
validation_train_kernels = validation_train_kernels.at[n_validation:].set(force_weight*validation_train_kernels[n_validation:])
validation_train_kernels = validation_train_kernels.at[:, n_train:].set(force_weight*validation_train_kernels[:, n_train:])
test_train_kernels = test_train_kernels.at[n_test:].set(force_weight*test_train_kernels[n_test:])
test_train_kernels = test_train_kernels.at[:, n_train:].set(force_weight*test_train_kernels[:, n_train:])

# 3D grid search
optimization_target = "MAE"
validation_score_best = 1e90
print((f"The optimization target is {optimization_target}"))
for log_C0 in range(6, 19):
    for log_C in range(-5, 8):
        for alpha in np.geomspace(0.001, 1.0, 101):
            C0 = 10**log_C0
            C = 10**log_C
            nu_coefficient_vector = jnp.array([C0] + [C*(alpha**nu)/factorial(nu) for nu in range(1, nu_max+1)])
            train_train_kernel = train_train_kernels @ nu_coefficient_vector
            validation_train_kernel = validation_train_kernels @ nu_coefficient_vector
            test_train_kernel = test_train_kernels @ nu_coefficient_vector
            c = jnp.linalg.solve(train_train_kernel+jnp.eye(train_train_kernel.shape[0]), train_targets)
            if optimization_target == "RMSE":
                train_error_forces = get_rmse(train_targets[n_train:], (train_train_kernel @ c)[n_train:]) / force_weight
                validation_error_forces = get_rmse(validation_targets[n_validation:], (validation_train_kernel @ c)[n_validation:]) / force_weight
                test_error_forces = get_rmse(test_targets[n_test:], (test_train_kernel @ c)[n_test:]) / force_weight
                train_error_energies = get_rmse(train_targets[:n_train], (train_train_kernel @ c)[:n_train])
                validation_error_energies = get_rmse(validation_targets[:n_validation], (validation_train_kernel @ c)[:n_validation])
                test_error_energies = get_rmse(test_targets[:n_test], (test_train_kernel @ c)[:n_test])
            elif optimization_target == "MAE":
                train_error_forces = get_mae(train_targets[n_train:], (train_train_kernel @ c)[n_train:]) / force_weight
                validation_error_forces = get_mae(validation_targets[n_validation:], (validation_train_kernel @ c)[n_validation:]) / force_weight
                test_error_forces = get_mae(test_targets[n_test:], (test_train_kernel @ c)[n_test:]) / force_weight
                train_error_energies = get_mae(train_targets[:n_train], (train_train_kernel @ c)[:n_train])
                validation_error_energies = get_mae(validation_targets[:n_validation], (validation_train_kernel @ c)[:n_validation])
                test_error_energies = get_mae(test_targets[:n_test], (test_train_kernel @ c)[:n_test])
            else:
                raise ValueError("The optimization target must be rmse or mae")
            #print()
            #print(log_C0, log_C, alpha)
            #print(train_error_energies, validation_error_energies, test_error_energies, train_error_forces, validation_error_forces, test_error_forces)
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
                c_best = c

print(f"Best optimization parameters: log_C0={log_C0_best}, log_C={log_C_best}, alpha={alpha_best}")
print(f"Best validation error: {validation_best_energies} {validation_best_forces}")

print(f"Test error ({optimization_target}):")
print(test_best_energies, test_best_forces)
print()"""

log_C0_best = 1
log_C_best = 1
alpha_best = 1
c_best = jnp.zeros((50*(1+train_structures[0].positions.shape[0]*3),))

C0_best = 10**log_C0_best
C_best = 10**log_C_best
nu_coefficient_vector_best = jnp.array([C0_best] + [C_best*(alpha_best**nu)/factorial(nu) for nu in range(1, nu_max+1)])


from utils.wigner_kernels import compute_wks_single_batch, compute_wks_single_batch_jac_2
from utils.dataset_processing import create_jax_structures

# Train and test ( is one single, test many)
jax_batch_evaluate = create_jax_structures([test_structures[0]], all_species, r_cut)  # to be changed to multiple ones?
jax_batch_train = create_jax_structures(train_structures, all_species, r_cut)

print("Finished pre-processing")

@partial(jax.jit, static_argnames=["n_train", "all_species", "l_max", "n_max", "nu_max"])
def compute_wks_single_batch_and_contract_over_nu(positions, train_positions, jax_structure_evaluate, jax_structures_train, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best):
    wigner_kernels = compute_wks_single_batch(positions, train_positions, jax_structure_evaluate, jax_structures_train,
        1, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines)
    return (wigner_kernels @ nu_coefficient_vector_best).squeeze(axis=0)

def compute_wks_single_batch_and_contract_over_nu_sum(positions, train_positions, jax_structure_evaluate, jax_structures_train, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best):
    contracted_kernels = compute_wks_single_batch_and_contract_over_nu(positions, train_positions, jax_structure_evaluate, jax_structures_train, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best)
    return jnp.sum(contracted_kernels, axis=0)

compute_wks_single_batch_and_contract_over_nu_jac = jax.jit(
    jax.grad(compute_wks_single_batch_and_contract_over_nu_sum, argnums=1),
    static_argnames=["n_train", "all_species", "l_max", "n_max", "nu_max"]
)

@partial(jax.jit, static_argnames=["n_train", "all_species", "l_max", "n_max", "nu_max"])
def evaluate_energies(positions, jax_structure_evaluate, jax_structures_train, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best, c_best):
    support_points = compute_wks_single_batch_and_contract_over_nu(positions, jax_structures_train["positions"], jax_structure_evaluate, jax_structures_train,
        n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best)
    support_points_derivatives = compute_wks_single_batch_and_contract_over_nu_jac(positions, jax_structures_train["positions"], jax_structure_evaluate, jax_structures_train,
        n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best)
    support_points_derivatives = support_points_derivatives.flatten()
    total_support_points = jnp.concatenate((support_points, support_points_derivatives))
    energy = total_support_points @ c_best
    return energy

evaluate_energies_and_forces = jax.jit(
    jax.value_and_grad(evaluate_energies),
    static_argnames=["n_train", "all_species", "l_max", "n_max", "nu_max"]
)

e, neg_f = evaluate_energies_and_forces(jax_batch_evaluate["positions"], jax_batch_evaluate, jax_batch_train, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best, c_best)
neg_f.block_until_ready()

import time
start = time.time()
for _ in range(1000):
    e, neg_f = evaluate_energies_and_forces(jax_batch_evaluate["positions"], jax_batch_evaluate, jax_batch_train, n_train, all_species, l_max, n_max, nu_max, cgs, radial_splines, nu_coefficient_vector_best, c_best)
    neg_f.block_until_ready()
finish = time.time()
print(f"Took {(finish-start)/1000}s")
