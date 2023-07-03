import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ase
from ase import io
from math import factorial

from utils.dataset_processing import JaxStructures
from utils.nu0_kernels import compute_wk_nu0
from utils.nu1_kernels import compute_wk_nu1
from utils.structure_wise import compute_stucture_wise_kernels
from utils.clebsch_gordan import get_cg_coefficients
from utils.wigner_iterations import compute_wks_high_order
from utils.error_measures import get_mae, get_rmse


np.random.seed(0)
n_train = 15
n_validation = 5
n_test = 10
train_validation_structures = ase.io.read("datasets/gold.xyz", ":20")
test_structures = ase.io.read("datasets/gold.xyz", "20:30")
np.random.shuffle(train_validation_structures)
train_structures = train_validation_structures[:n_train]
validation_structures = train_validation_structures[n_train:]

train_targets = jnp.array([train_structure.info["elec. Free Energy [eV]"] for train_structure in train_structures])
validation_targets = jnp.array([validation_structure.info["elec. Free Energy [eV]"] for validation_structure in validation_structures])
test_targets = jnp.array([test_structure.info["elec. Free Energy [eV]"] for test_structure in test_structures])

# structures = [ase.Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
jax_structures_train = JaxStructures(train_structures)
jax_structures_validation = JaxStructures(validation_structures)
jax_structures_test = JaxStructures(test_structures)

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



def compute_wks(jax_structures1, jax_structures2):

    wks_nu0, s1_0, s2_0 = compute_wk_nu0(jax_structures1, jax_structures2, all_species)
    wks_nu1, s1, s2 = compute_wk_nu1(jax_structures1, jax_structures2, all_species, l_max, r_cut, C_s, lambda_s)

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

train_train_kernels = compute_wks(jax_structures_train, jax_structures_train)
validation_train_kernels = compute_wks(jax_structures_validation, jax_structures_train)
test_train_kernels = compute_wks(jax_structures_test, jax_structures_train)

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
