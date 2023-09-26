import jax.numpy as jnp

import numpy as np
import scipy as sp
from scipy.special import jv
from scipy.optimize import brentq

from scipy.special import spherical_jn as j_l
from scipy.special import spherical_in as i_l
from scipy.integrate import quadrature


def generate_splines(
    radial_basis,
    radial_basis_derivatives,
    max_index,
    cutoff_radius,
    requested_accuracy=1e-8
):
    """Spline generator for tabulated radial integrals.

    Besides some self-explanatory parameters, this function takes as inputs two
    functions, namely radial_basis and radial_basis_derivatives. These must be
    able to calculate the radial basis functions by taking n, l, and r as their
    inputs, where n and l are integers and r is a numpy 1-D array that contains
    the spline points at which the radial basis function (or its derivative)
    needs to be evaluated. These functions should return a numpy 1-D array
    containing the values of the radial basis function (or its derivative)
    corresponding to the specified n and l, and evaluated at all points in the
    r array. If specified, n_spline_points determines how many spline points
    will be used for each splined radial basis function. Alternatively, the user
    can specify a requested accuracy. Spline points will be added until either
    the relative error or the absolute error fall below the requested accuracy on
    average across all radial basis functions.
    """

    def value_evaluator_2D(positions):
        values = []
        for index in range(max_index):
            value = radial_basis(index, np.array(positions))
            values.append(value)
        values = np.array(values)
        values = values.T
        values = values.reshape(len(positions), max_index)
        return values

    def derivative_evaluator_2D(positions):
        derivatives = []
        for index in range(max_index):
            derivative = radial_basis_derivatives(index, np.array(positions))
            derivatives.append(derivative)
        derivatives = np.array(derivatives)
        derivatives = derivatives.T
        derivatives = derivatives.reshape(len(positions), max_index)
        return derivatives

    splines = initialize_splines(
        value_evaluator_2D,
        derivative_evaluator_2D,
        cutoff_radius,
        requested_accuracy
    )

    return splines


def initialize_splines(
    radial_basis,
    radial_basis_derivatives,
    cutoff_radius,
    requested_accuracy=1e-8,
):

    start = 0.0
    stop = cutoff_radius
    values_fn = radial_basis
    derivatives_fn = radial_basis_derivatives
    requested_accuracy = requested_accuracy

    # initialize spline with 11 points
    positions = np.linspace(start, stop, 11)
    spline_positions = positions
    spline_values = values_fn(positions)
    spline_derivatives = derivatives_fn(positions)

    number_of_custom_axes = len(spline_values.shape) - 1

    while True:
        n_intermediate_positions = len(spline_positions) - 1

        if n_intermediate_positions >= 50000:
            raise ValueError(
                "Maximum number of spline points reached. \
                There might be a problem with the functions to be splined"
            )

        half_step = (spline_positions[1] - spline_positions[0]) / 2
        intermediate_positions = np.linspace(
            start + half_step, stop - half_step, n_intermediate_positions
        )

        estimated_values = compute_spline(intermediate_positions, spline_positions, spline_values, spline_derivatives, number_of_custom_axes)
        new_values = values_fn(intermediate_positions)

        mean_absolute_error = np.mean(np.abs(estimated_values - new_values))
        mean_relative_error = np.mean(
            np.abs((estimated_values - new_values) / new_values)
        )

        if (
            mean_absolute_error < requested_accuracy
            or mean_relative_error < requested_accuracy
        ):
            break

        new_derivatives = derivatives_fn(intermediate_positions)

        concatenated_positions = np.concatenate(
            [spline_positions, intermediate_positions], axis=0
        )
        concatenated_values = np.concatenate(
            [spline_values, new_values], axis=0
        )
        concatenated_derivatives = np.concatenate(
            [spline_derivatives, new_derivatives], axis=0
        )

        sort_indices = np.argsort(concatenated_positions, axis=0)

        spline_positions = concatenated_positions[sort_indices]
        spline_values = concatenated_values[sort_indices]
        spline_derivatives = concatenated_derivatives[sort_indices]

    return spline_positions, spline_values, spline_derivatives


def compute_spline(positions, spline_positions, spline_values, spline_derivatives, number_of_custom_axes):
    
    x = positions
    delta_x = spline_positions[1] - spline_positions[0]
    n = (np.floor(x / delta_x)).astype(np.int32)

    t = (x - n * delta_x) / delta_x
    t_2 = t**2
    t_3 = t**3

    h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0
    h10 = t_3 - 2.0 * t_2 + t
    h01 = -2.0 * t_3 + 3.0 * t_2
    h11 = t_3 - t_2

    p_k = spline_values[n]
    p_k_1 = spline_values[n + 1]

    m_k = spline_derivatives[n]
    m_k_1 = spline_derivatives[n + 1]

    new_shape = (-1,) + (1,) * number_of_custom_axes
    h00 = h00.reshape(new_shape)
    h10 = h10.reshape(new_shape)
    h01 = h01.reshape(new_shape)
    h11 = h11.reshape(new_shape)

    interpolated_values = (
        h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1
    )

    return interpolated_values


def Jn(r, n):
    return (np.sqrt(np.pi/(2*r))*jv(n+0.5,r))
def Jn_zeros(n, nt):
    zerosj = np.zeros((n+1, nt), dtype=np.float64)
    zerosj[0] = np.arange(1,nt+1)*np.pi
    points = np.arange(1,nt+n+1)*np.pi
    racines = np.zeros(nt+n, dtype=np.float64)
    for i in range(1,n+1):
        for j in range(nt+n-i):
            foo = brentq(Jn, points[j], points[j+1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:nt] = racines[:nt]
    return (zerosj)

def get_spherical_bessel_zeros(l_max, n_max):
    z_ln = Jn_zeros(l_max, n_max)  # Spherical Bessel zeros
    z_nl = z_ln.T
    return z_nl

def get_LE_splines(l_max, n_max, a, CS, l_r, accuracy=1e-3):

    z_nl = get_spherical_bessel_zeros(l_max, n_max)

    def R_nl(n, l, r):
        return j_l(l, z_nl[n, l]*r/a)

    def N_nl(n, l):
        # Normalization factor for LE basis functions
        def function_to_integrate_to_get_normalization_factor(x):
            return j_l(l, x)**2 * x**2
        integral, _ = sp.integrate.quadrature(function_to_integrate_to_get_normalization_factor, 0.0, z_nl[n, l], miniter=100, maxiter=10000)
        return (1.0/z_nl[n, l]**3 * integral)**(-0.5)

    N_nl_precomputed = np.zeros((n_max, l_max+1))
    for l in range(l_max+1):
        for n in range(n_max):
            N_nl_precomputed[n, l] = N_nl(n, l)

    def get_LE_function(n, l, r):
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            R[i] = R_nl(n, l, r[i])
        return N_nl_precomputed[n, l]*R*a**(-1.5)

    def sigma(r):  
        # The function that determines how sigma changes as a function of r.
        sigma = CS*np.exp(r/l_r)
        return sigma

    from fortran import sbessi
    def exp_i_l(l, x):
        result = np.zeros_like(x)
        for i in range(len(x)):
            result[i] = sbessi(l, x[i])
        return result

    def evaluate_LE_function_mollified_adaptive(n, l, r):
        # Calculates a mollified (but with adaptive sigma) LE radial basis function for a signle value of r.
        c = 1.0/(2.0*sigma(r)**2)
        def function_to_integrate(x):
            return 4.0 * np.pi * x**2 * get_LE_function(n, l, x) * np.exp(-c*(x-r)**2) * exp_i_l(l, 2.0*c*x*r) * (1.0/(np.sqrt(2*np.pi)*sigma(r)))**3 # * (1.0/(np.pi*sigma(r)**2))**(3/4) #
        integral, _ = sp.integrate.quadrature(function_to_integrate, 0.0, a, miniter=100, maxiter=10000)
        return integral

    def get_LE_function_mollified_adaptive(n, l, r):
        # Calculates a mollified (but with adaptive sigma) LE radial basis function for a 1D array of values r.
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            if r[i] < 0.5:  # Reduce computational cost
                R[i] = evaluate_LE_function_mollified_adaptive(n, l, 0.5) * np.exp(-0.5/l_r)
            else:
                R[i] = evaluate_LE_function_mollified_adaptive(n, l, r[i]) * np.exp(-r[i]/l_r)
            # print(r[i], R[i])
        return R

    def function_for_splining(n, l, x):
        return get_LE_function_mollified_adaptive(n, l, x)

    def function_for_splining_derivative(n, l, r):
        delta = 1e-6
        all_derivatives_except_first_and_last = (function_for_splining(n, l, r[1:-1]+delta) - function_for_splining(n, l, r[1:-1]-delta)) / (2.0*delta)
        derivative_at_zero = (function_for_splining(n, l, np.array([delta/10.0])) - function_for_splining(n, l, np.array([0.0]))) / (delta/10.0)
        derivative_last = (function_for_splining(n, l, np.array([a])) - function_for_splining(n, l, np.array([a-delta/10.0]))) / (delta/10.0)
        return np.concatenate([derivative_at_zero, all_derivatives_except_first_and_last, derivative_last])

    def function_for_splining_wrapped(index, r):
        l = index // n_max
        n = index - l*n_max
        return function_for_splining(n, l, r)

    def function_for_splining_derivative_wrapped(index, r):
        l = index // n_max
        n = index - l*n_max
        return function_for_splining_derivative(n, l, r)

    return generate_splines(
        function_for_splining_wrapped,
        function_for_splining_derivative_wrapped,
        (l_max+1)*n_max,
        a,
        requested_accuracy=accuracy
    )


def compute_spline_jax(positions, spline_positions, spline_values, spline_derivatives, number_of_custom_axes):
    
    x = positions
    delta_x = spline_positions[1] - spline_positions[0]
    n = (jnp.floor(x / delta_x)).astype(jnp.int32)

    t = (x - n * delta_x) / delta_x
    t_2 = t**2
    t_3 = t**3

    h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0
    h10 = t_3 - 2.0 * t_2 + t
    h01 = -2.0 * t_3 + 3.0 * t_2
    h11 = t_3 - t_2

    p_k = spline_values[n]
    p_k_1 = spline_values[n + 1]

    m_k = spline_derivatives[n]
    m_k_1 = spline_derivatives[n + 1]

    new_shape = (-1,) + (1,) * number_of_custom_axes
    h00 = h00.reshape(new_shape)
    h10 = h10.reshape(new_shape)
    h01 = h01.reshape(new_shape)
    h11 = h11.reshape(new_shape)

    interpolated_values = (
        h00 * p_k + h10 * delta_x * m_k + h01 * p_k_1 + h11 * delta_x * m_k_1
    )

    return interpolated_values


if __name__ == "__main__":

    positions, values, derivatives = get_LE_splines(3, 25, 6.0, 0.2, 3.0)
    # (l_max, n_max, a, CS, l_r)
    # to be tested

    import matplotlib.pyplot as plt
    x = np.linspace(0, 5.9, 1000)
    y = compute_spline(x, positions, values, derivatives, 1)
    plt.plot(x, y[:, 53])
    plt.savefig("plot.pdf")


