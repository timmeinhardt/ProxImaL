from __future__ import print_function
from proximal.lin_ops import (CompGraph, est_CompGraph_norm, Variable,
                              vstack)
from proximal.utils.timings_log import TimingsLog, TimingsEntry
from .invert import get_least_squares_inverse, max_diag_set
import numpy as np


def partition(prox_fns, try_diagonalize=True):
    """Divide the proxable functions into sets Psi and Omega.
    """
    # Omega must be a single function.
    # Merge quadratic functions into the x update.
    # Automatically try to split the problem.
    quad_fns = max_diag_set(prox_fns)
    split_fn = []
    omega_fns = []
    if len(quad_fns) == 0:
        for fn in prox_fns:
            if type(fn.lin_op) == Variable:
                split_fn = [fn]
                break
        omega_fns = split_fn
    else:
        # Proximal solve for:
        # G(x) + 1/(2*tau) * ||x - v||^2_2, with G containing all quadratics
        quad_ops = []
        const_terms = []
        alpha_terms = []
        for fn in quad_fns:
            #fn = fn.absorb_params()
            quad_ops.append(fn.alpha * fn.beta * fn.lin_op)
            const_terms.append(fn.b.flatten())
            alpha_terms.append(fn.alpha)

        b = np.hstack(const_terms)
        alphas = np.hstack(alpha_terms)
        # Get optimize inverse (tries spatial and frequency diagonalization)
        # x_update = get_least_squares_inverse(quad_ops, b, try_diagonalize)
        omega_fns = [vstack(quad_ops), b, alphas]

    psi_fns = [func for func in prox_fns if func not in split_fn + quad_fns]
    return psi_fns, omega_fns


def solve(psi_fns, omega_fns, tau=None, alpha=1.0,
          max_iters=1000, eps=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, try_fast_norm=False, scaled=True,
          metric=None, convlog=None, verbose=0, score_func=None,
          Knorm=None):
    # omega consists of squared terms and constant terms like f.
    assert len(omega_fns) <= 3
    prox_fns = psi_fns + omega_fns[0:1]
    K = CompGraph(omega_fns[0])
    v = np.zeros(K.input_size)

    tau = est_CompGraph_norm(K, try_fast_norm)**2

    # Initialize
    x = np.zeros(K.input_size)
    y = np.zeros(K.output_size)

    if x0 is not None:
        x[:] = np.reshape(x0, K.input_size)
        K.forward(x, y)

    # Buffers.
    Kx = np.zeros(K.output_size)
    KTKx = np.zeros(K.input_size)

    prev_x = x.copy()
    prev_Kx = Kx.copy()
    prev_y = y.copy()

    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("FBS iteration")

    # Convergence log for initial iterate
    if convlog is not None:
        K.update_vars(x)
        objval = sum([fn.value for fn in prox_fns])
        convlog.record_objective(objval)
        convlog.record_timing(0.0)

    score = 0
    for i in range(max_iters):
        iter_timing.tic()
        if convlog is not None:
            convlog.tic()

        # Keep track of previous iterates
        np.copyto(prev_x, x)
        np.copyto(prev_Kx, Kx)
        np.copyto(prev_y, y)

        # Compute x
        K.forward(x, Kx)
        K.adjoint(Kx - omega_fns[1], KTKx)
        x = x - tau * omega_fns[2] * KTKx

        # Update y.
        offset = 0
        for fn in psi_fns:
            slc = slice(offset, offset + fn.lin_op.size, None)
            x_slc = np.reshape(x[slc], fn.lin_op.shape)

            prox_log[fn].tic()
            y[slc] = fn.prox(1.0, x_slc, i, verbose=verbose, score_func=score_func).flatten()
            prox_log[fn].toc()
            offset += fn.lin_op.size
        y[offset:] = 0
        # Update x

        x = alpha * x + (1 - alpha) * y

        # Convergence log
        if convlog is not None:
            convlog.toc()
            K.update_vars(x)
            objval = sum([fn.value for fn in prox_fns])
            convlog.record_objective(objval)

        # Check convergence
        r_x = np.linalg.norm(x - prev_x)
        r_y = np.linalg.norm(y - prev_y)
        error = r_x + r_y

        if score_func is not None:
            x_now = x.copy()
            if scaled:
                x_now /= np.sqrt(Knorm)
            prev_score = score
            score = score_func(x_now)

        # Progress
        if scaled is not None and verbose > 0:
            print("it %i psnr %f error %f" % (i, score, error))

        if score_func is not None and prev_score > score or error <= eps:
            break

        iter_timing.toc()

    # Print out timings info.
    if verbose > 0:
        print(iter_timing)
        print("prox funcs:")
        print(prox_log)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    K.update_vars(x)

    # Return optimal value.
    return sum([fn.value for fn in prox_fns])


def est_params_pc(K, tau=None, sigma=None, verbose=True, scaled=False, try_fast_norm=False):

    # Select theta constant and sigma larger 0
    theta = 1.0
    sigma = 1.0 if sigma is None else sigma

    # Estimate Lipschitz constant and comput tau
    if scaled:
        L = 1
    else:
        L = est_CompGraph_norm(K, try_fast_norm)
    tau = 1.0 / (sigma * L**2)

    if verbose:
        print("Estimated params [sigma = %3.3f | tau = %3.3f | theta = %3.3f | L_est = %3.4f]"
              % (sigma, tau, theta, L))

    return tau, sigma, theta
