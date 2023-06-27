import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=1)
def spherical_harmonics(xyz, l_max):
    # Real, cartesian spherical harmonics (normalized)
    assert len(xyz.shape) == 2
    assert xyz.shape[1] == 3
    r = jnp.sqrt(jnp.sum(xyz**2, axis=1, keepdims=True))
    xyz_normalized = xyz/r
    prefactors = jnp.empty(((l_max+1)**2,))
    ylm = jnp.empty(xyz.shape[:-1] + ((l_max+1)**2,))
    for l in range(l_max+1):
        prefactors = prefactors.at[l**2+l].set(jnp.sqrt((2*l+1)/(2*jnp.pi)))
        for m in range(1, l+1):
            prefactors = prefactors.at[l**2+l+m].set(-prefactors[l**2+l+m-1]/jnp.sqrt((l+m)*(l-m+1)))
            prefactors = prefactors.at[l**2+l-m].set(-prefactors[l**2+l-m+1]/jnp.sqrt((l+m)*(l-m+1)))
        if l == 0:
            ylm = ylm.at[:, 0].set(1/jnp.sqrt(2.0))
        elif l == 1:
            ylm = ylm.at[:, 1].set(-xyz_normalized[:, 1])
            ylm = ylm.at[:, 2].set(xyz_normalized[:, 2]/jnp.sqrt(2.0))
            ylm = ylm.at[:, 3].set(-xyz_normalized[:, 0])
        else:
            pos_m_range = jnp.arange(1, l-1)
            neg_m_range = jnp.arange(-1, -l+1, -1)
            ylm = ylm.at[:, l**2].set(-(2*l-1)*(ylm[:, (l-1)**2]*xyz_normalized[:, 0]+ylm[:, (l-1)**2+2*(l-1)]*xyz_normalized[:, 1]))
            ylm = ylm.at[:, l**2+2*l].set(-(2*l-1)*(-ylm[:, (l-1)**2]*xyz_normalized[:, 1]+ylm[:, (l-1)**2+2*(l-1)]*xyz_normalized[:, 0]))
            ylm = ylm.at[:, l**2+1].set((2*l-1)*xyz_normalized[:, 2]*ylm[:, (l-1)**2])
            ylm = ylm.at[:, l**2+2*l-1].set((2*l-1)*xyz_normalized[:, 2]*ylm[:, (l-1)**2+2*(l-1)])
            ylm = ylm.at[:, l**2+l].set(((2*l-1)*xyz_normalized[:, 2]*ylm[:, (l-1)**2+(l-1)]-(l-1)*ylm[:, (l-2)**2+(l-2)])/l)
            ylm = ylm.at[:, l**2+l+pos_m_range].set(((2*l-1)*xyz_normalized[:, [2]]*ylm[:, (l-1)**2+(l-1)+pos_m_range]-(l+pos_m_range-1)*ylm[:, (l-2)**2+(l-2)+pos_m_range])/(l-pos_m_range))
            ylm = ylm.at[:, l**2+l+neg_m_range].set(((2*l-1)*xyz_normalized[:, [2]]*ylm[:, (l-1)**2+(l-1)+neg_m_range]-(l-neg_m_range-1)*ylm[:, (l-2)**2+(l-2)+neg_m_range])/(l+neg_m_range))
    return prefactors*ylm


@partial(jax.jit, static_argnums=4)
def asymptotic_formula(twox, on2x, s, rmu, l):
    s = 1.0
    t = 1.0
    u = 0.5
    v = twox
    for _ in range(1, l+1):
        t = -(rmu-u**2)*t/v
        s = s+t
        u = u+1.0
        v = v+twox
    s = on2x*s
    sbessi = s
    return sbessi
vaux1 = jax.vmap(asymptotic_formula, (0, 0, 0, None, None))


@partial(jax.jit, static_argnums=2)
def downward_recursion(on2x, s, l):
    s0 = s
    m = l+30
    u = 4*on2x
    t = (m+1.5)*u
    sp = 0.0
    s = 1.0
    for _ in range(m, l, -1):
        t = t-u
        sn = t*s+sp
        sp = s/sn
        s = 1.0
        m -= 1.0
    for _ in range(l, 0, -1):
        t = t-u
        sn = t*s+sp
        sp = s
        s = sn
    sbessi = s0/s
    return sbessi
vaux2 = jax.vmap(downward_recursion, (0, 0, None))


@partial(jax.jit, static_argnums=1)
def scaled_spherical_bessel_i(x, l):
    # Scaled Modified Spherical Bessel function s = exp(-x)*i_l(x)
    # Due to the requirement of jax compilation, it might fail for some edge cases.
    # One of them is x = 0

    twox = 2.0*x
    on2x = 1.0/twox
    rmu = (l+0.5)**2
    sbessi = jnp.where(
        x > max(19, rmu),
        asymptotic_formula(2.0*x, 1/(2.0*x), jnp.where(x < 19, (1.0-jnp.exp(-2.0*x))*1/(2.0*x), 1/(2.0*x)), rmu, l),  # Asymptotic formula for large x
        downward_recursion(on2x, jnp.where(x < 19, (1.0-jnp.exp(-2.0*x))*1/(2.0*x), 1/(2.0*x)), l)  # Downward recursion for small x
    )
    return sbessi
