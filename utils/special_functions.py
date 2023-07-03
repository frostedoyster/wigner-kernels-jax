import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.jit, static_argnums=(1,))
def spherical_harmonics(xyz, l_max):
    # Spherical harmonics

    assert xyz.shape[0] == 3
    r = jnp.sqrt(jnp.sum(xyz**2, keepdims=True))
    xyz_normalized = xyz/r
    prefactors = jnp.empty(((l_max+1)**2,))
    ylm = jnp.empty(((l_max+1)**2,))

    for l in range(l_max+1):
        prefactors = prefactors.at[l**2+l].set(jnp.sqrt((2*l+1)/(2*jnp.pi)))
        for m in range(1, l+1):
            prefactors = prefactors.at[l**2+l+m].set(-prefactors[l**2+l+m-1]/jnp.sqrt((l+m)*(l-m+1)))
            prefactors = prefactors.at[l**2+l-m].set(-prefactors[l**2+l-m+1]/jnp.sqrt((l+m)*(l-m+1)))
        if l == 0:
            ylm = ylm.at[0].set(1/jnp.sqrt(2.0))
        elif l == 1:
            ylm = ylm.at[1].set(-xyz_normalized[1])
            ylm = ylm.at[2].set(xyz_normalized[2]/jnp.sqrt(2.0))
            ylm = ylm.at[3].set(-xyz_normalized[0])
        else:
            pos_m_range = jnp.arange(1, l-1)
            neg_m_range = jnp.arange(-1, -l+1, -1)
            ylm = ylm.at[l**2].set(-(2*l-1)*(ylm[(l-1)**2]*xyz_normalized[0]+ylm[(l-1)**2+2*(l-1)]*xyz_normalized[1]))
            ylm = ylm.at[l**2+2*l].set(-(2*l-1)*(-ylm[(l-1)**2]*xyz_normalized[1]+ylm[(l-1)**2+2*(l-1)]*xyz_normalized[0]))
            ylm = ylm.at[l**2+1].set((2*l-1)*xyz_normalized[2]*ylm[(l-1)**2])
            ylm = ylm.at[l**2+2*l-1].set((2*l-1)*xyz_normalized[2]*ylm[(l-1)**2+2*(l-1)])
            ylm = ylm.at[l**2+l].set(((2*l-1)*xyz_normalized[2]*ylm[(l-1)**2+(l-1)]-(l-1)*ylm[(l-2)**2+(l-2)])/l)
            ylm = ylm.at[l**2+l+pos_m_range].set(((2*l-1)*xyz_normalized[2]*ylm[(l-1)**2+(l-1)+pos_m_range]-(l+pos_m_range-1)*ylm[(l-2)**2+(l-2)+pos_m_range])/(l-pos_m_range))
            ylm = ylm.at[l**2+l+neg_m_range].set(((2*l-1)*xyz_normalized[2]*ylm[(l-1)**2+(l-1)+neg_m_range]-(l-neg_m_range-1)*ylm[(l-2)**2+(l-2)+neg_m_range])/(l+neg_m_range))

    return prefactors*ylm


@partial(jax.jit, static_argnums=1)
@partial(jax.vmap, in_axes=(0, None))
def scaled_spherical_bessel_i(x, l_max):
    # Scaled Modified Spherical Bessel function s = exp(-x)*i_l(x)
    # Not to be used for very small x

    sbessi = jnp.empty((l_max+1,))
    for l in range(l_max+1):
        if l == 0:
            sbessi = sbessi.at[l].set(
                (1.0-jnp.exp(-2.0*x))/(2.0*x)
            )
        elif l == 1:
            sbessi = sbessi.at[l].set(
                (1.0+jnp.exp(-2.0*x) - (1.0-jnp.exp(-2.0*x))/x)/(2.0*x)
            )
        else:
            sbessi = sbessi.at[l].set(
                -(2*l-1)*sbessi[l-1]/x + sbessi[l-2]
            )
    
    return sbessi
