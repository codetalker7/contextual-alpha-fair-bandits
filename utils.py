import numpy as np

def projectOnSimplex(v):
    n, = v.shape
    # check if we are already on the simplex
    if v.sum() == 1 and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - 1) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def log_opt(v):
    """
    Solves the maximization problem sum_i log(x_i) + sum_i x_i * z_i
    subject to the constraint that x lies on the standard simplex
    """

    est = -v.max() - 1
    f_val = 1

    while(abs(f_val) >= 1e-9 or max(v + est) > 0):
        f_val = (1 / (v + est)).sum() + 1
        f_dash = -(((1 / (v + est))**2).sum())
        est = est - f_val / f_dash

    opt_point = -(1 / (v + est))
    opt_val = np.log(opt_point).sum() + np.inner(opt_point, v)
    return (opt_point, opt_val)


