import numpy as np

def prox_linf(x, alpha):
    """
    Proximal operator for L∞ norm
    prox_linf(x, alpha) = x - proj_l1_ball(x, alpha)
    """
    if alpha < 0:
        raise ValueError("alpha should be positive")

    return x - proj_l1_ball(x, alpha)


def proj_l1_ball(x, r=1):
    """
    Projection onto L1 ball
    proj_l1_ball(x, r) = sign(x) * proj_simplex(|x|, r, 'ineq')
    """
    if r < 0:
        raise ValueError("r should be non-negative")

    return np.sign(x) * proj_simplex(np.abs(x), r, 'ineq')


def proj_simplex(x, r=1, eq_flag='eq'):
    """
    Projection onto simplex
    """
    if r < 0:
        raise ValueError("Set is infeasible")

    if eq_flag == 'eq':
        # Projection onto simplex with equality constraint
        return proj_hyperplane_box(x, np.ones_like(x), r, 0)
    elif eq_flag == 'ineq':
        # Projection onto simplex with inequality constraint
        return proj_halfspace_box(x, np.ones_like(x), r, 0)
    else:
        raise ValueError("eq_flag should be either 'eq' or 'ineq'")


def proj_hyperplane_box(x, a, b, l=-1000, u=1000):
    """
    Compute the orthogonal projection of point x onto the intersection of a hyperplane and box constraints
    {x : <a,x> = b, l <= x <= u}

    Parameters:
    x - point to project (vector/matrix)
    a - vector/matrix
    b - scalar
    l - lower bound (vector/matrix/scalar) [default: -inf]
    u - upper bound (vector/matrix/scalar) [default: inf]

    Assumptions:
    The intersection of the hyperplane and box constraints is non-empty

    Returns:
    out - projected vector
    """
    # Handle boundary cases
    if np.isscalar(l):
        l = np.full_like(x, l)
    if np.isscalar(u):
        u = np.full_like(x, u)

    # Compute sumlb and sumub
    # sumlb = trace(a'*((l .* (sign(a)>0)) + (u .* (sign(a)<0))))
    # sumub = trace(a'*((u .* (sign(a)>0)) + (l .* (sign(a)<0))))

    # Create sign masks
    sign_pos = a > 0
    sign_neg = a < 0

    # Compute sumlb
    term1_lb = l * sign_pos
    term2_lb = u * sign_neg
    sumlb = np.trace(a.T @ (term1_lb + term2_lb))

    # Compute sumub
    term1_ub = u * sign_pos
    term2_ub = l * sign_neg
    sumub = np.trace(a.T @ (term1_ub + term2_ub))

    # Check feasibility
    if sumlb > b or np.any(l > u) or sumub < b:
        raise ValueError("Set is infeasible")

    # Define function f(λ) = trace(a'*min(max(x-λ*a,l),u)) - b
    eps = 1e-10

    def f(lam):
        x_lam_a = x - lam * a
        clamped = np.minimum(np.maximum(x_lam_a, l), u)
        return np.trace(a.T @ clamped) - b

    # Find a suitable search interval
    lambda_min = -1
    while f(lambda_min) < 0:
        lambda_min *= 2

    lambda_max = 1
    while f(lambda_max) > 0:
        lambda_max *= 2

    # Solve using bisection method
    final_lam = bisection(f, lambda_min, lambda_max, eps)

    # Compute final projection
    out = np.minimum(np.maximum(x - final_lam * a, l), u)

    return out

def proj_halfspace_box(x, a, b, lb=-1000, ub=1000):
    """
    Compute the orthogonal projection of point x onto the intersection of a half-space and box constraints
    {x : <a,x> <= b, lb <= x <= ub}

    Parameters:
    x - point to project (vector/matrix)
    a - vector/matrix
    b - scalar
    lb - lower bound (vector/matrix/scalar) [default: -inf]
    ub - upper bound (vector/matrix/scalar) [default: inf]

    Assumptions:
    The intersection of the half-space and box constraints is non-empty

    Returns:
    out - projected vector
    """
    # Handle boundary cases
    if np.isscalar(lb):
        lb = np.full_like(x, lb)
    if np.isscalar(ub):
        ub = np.full_like(x, ub)

    # Check if lower bound exceeds upper bound
    if np.any(lb > ub):
        raise ValueError("Set is infeasible")

    # Project x onto box constraints
    x_box = np.minimum(np.maximum(x, lb), ub)

    # Compute <a, x_box>
    if x_box.ndim == 1:
        a_dot_xbox = np.dot(a, x_box)
    else:
        a_dot_xbox = np.trace(a.T @ x_box)

    # Check if already in feasible region
    if a_dot_xbox <= b:
        out = x_box
    else:
        # Otherwise project onto the boundary
        out = proj_hyperplane_box(x, a, b, lb, ub)

    return out


def bisection(f, min_val, max_val, eps=1e-10):
    """
    Bisection method to solve equation f(x) = 0
    """
    saved_fmin_val = f(min_val)
    saved_fmax_val = f(max_val)

    if saved_fmin_val * saved_fmax_val > 0:
        raise ValueError("f(lb)*f(ub) > 0")

    if min_val > max_val:
        raise ValueError("minimal value is bigger than maximal_value")

    iter_count = 0
    changed_limits = False
    r = None

    while max_val - min_val > eps:
        # If the function is linear in this interval, solve directly
        if abs(saved_fmin_val - saved_fmax_val) > eps:
            r = (max_val - (saved_fmax_val / saved_fmin_val) * min_val) / (1 - (saved_fmax_val / saved_fmin_val))
            saved_froot = f(r)

            if abs(saved_froot) < eps:
                return r

        iter_count += 1
        mid = (min_val + max_val) / 2
        changed_limits = False

        # Check if r can be used as a new bound
        if r is not None and r > mid:
            # r may become a new lower bound
            if saved_froot * saved_fmax_val < 0:
                min_val = r
                saved_fmin_val = saved_froot
                changed_limits = True
        elif r is not None:
            # r may become a new upper bound
            if saved_froot * saved_fmin_val < 0:
                max_val = r
                saved_fmax_val = saved_froot
                changed_limits = True

        if not changed_limits:
            # r not used, use midpoint instead
            saved_fmid = f(mid)
            if saved_fmin_val * saved_fmid > 0:
                min_val = mid
                saved_fmin_val = saved_fmid
            else:
                max_val = mid
                saved_fmax_val = saved_fmid

    return r if changed_limits and r is not None else mid


def prox_l1(x, alpha):
    """
    Proximal operator for L1 norm
    prox_l1(x, alpha) = max(|x| - alpha, 0) * sign(x)

    Parameters:
    x - point to project (vector/matrix)
    alpha - positive scalar

    Returns:
    out - proximal operator at x
    """
    if alpha < 0:
        raise ValueError("alpha should be positive")

    return np.maximum(np.abs(x) - alpha, 0) * np.sign(x)


def proj_affine_set_A(x, A, b):
    """
    Compute the orthogonal projection of point x onto the affine set {x: Ax = b}

    Parameters:
    x - point to project (vector)
    A - mxn matrix
    b - vector of length m

    Assumptions:
    A has full row rank

    Returns:
    out - projected vector
    """
    # Compute (A*A') \ (A*x - b)
    # Solve linear system using numpy
    ATA = A @ A.T
    Ax_minus_b = A @ x - b
    lambda_val = np.linalg.solve(ATA, Ax_minus_b)

    # Compute projection
    out = x - A.T @ lambda_val

    return out


def objectiveR2(lambda_val, A, b, xbar):
    """
    Compute objective function value: ||Ax + b||∞ + lambda*||x||₁

    Parameters:
    lambda_val - regularization parameter
    A - coefficient matrix
    b - constant term vector
    xbar - variable vector

    Returns:
    obj - objective function value
    """
    # Compute ||Ax + b||∞ (L-infinity norm, max norm)
    Ax_plus_b = A @ xbar + b
    linf_norm = np.linalg.norm(Ax_plus_b, ord=np.inf)

    # Compute lambda*||x||₁ (L1 norm, sum of absolute values)
    l1_norm = lambda_val * np.linalg.norm(xbar, ord=1)

        # Objective function value
    obj = linf_norm + l1_norm

    return obj


def ADMM_Sbeta(lambda_val, A, b):
    """
    Solve optimization problem using ADMM: min ||Ax + b||∞ + lambda*||x||₁

    Parameters:
    lambda_val - regularization parameter
    A - mxn coefficient matrix
    b - constant term vector of length m

    Returns:
    x2 - optimal solution
    history - iteration history
    """
    m, n = A.shape

    # Initialize variables
    z1 = np.zeros((m, 1))
    z2 = np.zeros((n, 1))
    z = np.concatenate((z1, z2))

    u1 = np.zeros((m, 1))
    u2 = np.zeros((n, 1))
    u = np.concatenate((u1, u2))

    QUIET = 1
    ABSTOL = 1e-6
    RELTOL = 1e-5

    # Define Atu and btu
    Atu = np.hstack((np.eye(m), -A))
    btu = b.reshape(-1, 1)  # Ensure b is a column vector

    # Define prox4 as prox_l1 function
    prox4 = lambda x: prox_l1(x, lambda_val)

    max_iter = 10000

    # Initialize history records
    history = {
        'objval': np.zeros(max_iter),
        'r_norm': np.zeros(max_iter),
        's_norm': np.zeros(max_iter),
        'eps_pri': np.zeros(max_iter),
        'eps_dual': np.zeros(max_iter)
    }

    for k in range(max_iter):
        # Update x1 and x2
        u1 = u[:m]
        z1 = z[:m]
        x1 = prox_linf(z1 - u1, 1)

        u2 = u[m:m + n]
        z2 = z[m:m + n]
        x2 = prox4(z2 - u2)

        x = np.concatenate((x1, x2))
        u_current = np.concatenate((u1, u2))

        # Save old z value
        zold = z.copy()

        # Project onto affine set
        z = proj_affine_set_A(x + u_current, Atu, btu)

        # Update u
        u = u_current + x - z

        # Compute objective function value
        history['objval'][k] = objectiveR2(lambda_val, A, b, x2)

        # Compute residuals
        history['r_norm'][k] = np.linalg.norm(x - z)
        history['s_norm'][k] = np.linalg.norm(zold - z)

        # Compute convergence tolerance
        p = len(x)
        history['eps_pri'][k] = np.sqrt(p) * ABSTOL + RELTOL * max(np.linalg.norm(x), np.linalg.norm(-z))
        history['eps_dual'][k] = np.sqrt(p) * ABSTOL + RELTOL * np.linalg.norm(u)

        # Print iteration info
        if not QUIET:
            print(f'{k + 1:3d}\t{history["r_norm"][k]:10.4f}\t{history["eps_pri"][k]:10.4f}\t'
                  f'{history["s_norm"][k]:10.4f}\t{history["eps_dual"][k]:10.4f}\t{history["objval"][k]:10.2f}')

        # Check convergence conditions
        if (history['r_norm'][k] < history['eps_pri'][k] and
                history['s_norm'][k] < history['eps_dual'][k]):
            # Truncate history records
            history['objval'] = history['objval'][:k + 1]
            history['r_norm'] = history['r_norm'][:k + 1]
            history['s_norm'] = history['s_norm'][:k + 1]
            history['eps_pri'] = history['eps_pri'][:k + 1]
            history['eps_dual'] = history['eps_dual'][:k + 1]
            break

    return x2, history


# Test code
if __name__ == "__main__":
    print("=== Test ADMM_Sbeta Algorithm ===")

    # Create test problem
    np.random.seed(42)
    m, n = 10, 20  # Number of rows and columns
    A = np.random.randn(m, n)
    x_true = np.random.randn(n, 1)
    # Make x_true sparse
    x_true[np.abs(x_true) < 1] = 0
    b = -A @ x_true + 0.1 * np.random.randn(m, 1)  # Add some noise
    lambda_val = 0.1

    print(f"Problem size: A ∈ R^{m}×{n}, b ∈ R^{m}")
    print(f"lambda = {lambda_val}")

    # Run ADMM algorithm
    x_sol, history = ADMM_Sbeta(lambda_val, A, b)

    # Print results
    print(f"\nNumber of iterations: {len(history['objval'])}")
    print(f"Final objective function value: {history['objval'][-1]:.4f}")
    print(f"Primal residual: {history['r_norm'][-1]:.4e}")
    print(f"Dual residual: {history['s_norm'][-1]:.4e}")

    # Compare true solution and numerical solution
    print(f"\nNumber of non-zero elements in true solution: {np.sum(np.abs(x_true) > 1e-6)}")
    print(f"Number of non-zero elements in numerical solution: {np.sum(np.abs(x_sol) > 1e-6)}")

    # Compute relative error
    if np.linalg.norm(x_true) > 1e-10:
        rel_error = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
        print(f"Relative error: {rel_error:.4e}")

    # Plot convergence curves
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Objective function value
        plt.subplot(2, 2, 1)
        plt.plot(history['objval'])
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('Objective value convergence curve')
        plt.grid(True)

        # Primal residual
        plt.subplot(2, 2, 2)
        plt.semilogy(history['r_norm'], label='Primal residual')
        plt.semilogy(history['eps_pri'], label='Primal residual tolerance')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Primal residual convergence curve')
        plt.legend()
        plt.grid(True)

        # Dual residual
        plt.subplot(2, 2, 3)
        plt.semilogy(history['s_norm'], label='Dual residual')
        plt.semilogy(history['eps_dual'], label='Dual residual tolerance')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Dual residual convergence curve')
        plt.legend()
        plt.grid(True)

        # Solution comparison
        plt.subplot(2, 2, 4)
        plt.plot(x_true, 'b-', label='True solution')
        plt.plot(x_sol, 'r--', label='Numerical solution')
        plt.xlabel('Variable index')
        plt.ylabel('Variable value')
        plt.title('True solution vs numerical solution comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nWarning: matplotlib not installed, cannot plot convergence curves")