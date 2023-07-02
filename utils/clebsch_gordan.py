import jax
import jax.numpy as jnp
import wigners


def get_cg_coefficients(l_max):

    cg_dictionary = {}
    for l1 in range(l_max+1):
        for l2 in range(l_max+1):
            for L in range(abs(l2-l1), min(l_max, l1+l2)+1):
                add_cg(cg_dictionary, l1, l2, L)

    return cg_dictionary


def add_cg(cg_dictionary, l1, l2, L):
    # print(f"Adding new CGs with l1={l1}, l2={l2}, L={L}")

    if (l1, l2, L) in cg_dictionary: 
        raise ValueError("Trying to add CGs that are already present... exiting")

    maxx = max(l1, max(l2, L))

    # real-to-complex and complex-to-real transformations as matrices
    r2c = {}
    c2r = {}
    for l in range(0, maxx + 1):
        r2c[l] = real2complex(l)
        c2r[l] = jnp.conjugate(r2c[l]).T

    complex_cg = complex_clebsch_gordan_matrix(l1, l2, L)

    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
        complex_cg.shape
    )

    real_cg = real_cg.swapaxes(0, 1)
    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
        real_cg.shape
    )
    real_cg = real_cg.swapaxes(0, 1)

    real_cg = real_cg @ c2r[L].T

    if (l1 + l2 + L) % 2 == 0:
        rcg = jnp.real(real_cg)
    else:
        rcg = jnp.imag(real_cg)

    cg_dictionary[(l1, l2, L)] = rcg


def real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = jnp.zeros((2 * L + 1, 2 * L + 1), dtype=jnp.complex128)

    I_SQRT_2 = 1.0 / jnp.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result = result.at[L - m, L + m].set(I_SQRT_2 * 1j * (-1) ** m)
            result = result.at[L + m, L + m].set(-I_SQRT_2 * 1j)

        if m == 0:
            result = result.at[L, L].set(1.0)

        if m > 0:
            result = result.at[L + m, L + m].set(I_SQRT_2 * (-1) ** m)
            result = result.at[L - m, L + m].set(I_SQRT_2)

    return result


def complex_clebsch_gordan_matrix(l1, l2, L):
    if jnp.abs(l1 - l2) > L or jnp.abs(l1 + l2) < L:
        raise RuntimeError(f"Called invalid CG: l1={l1}, l2={l2}, L={L}")
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)
