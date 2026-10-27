import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
f = h5py.File(os.path.join(here, 'data.hdf'), 'r')

# f = h5py.File('./data.hdf', 'r')

print(f.keys())

print(f['data'].keys())

xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

# plt.scatter(xpos, ypos)
# plt.show()

# sort by x, in case the data is out of order
order = np.argsort(xpos)
xpos, ypos = xpos[order], ypos[order]

# check the spacing assumption for the FFT
dx = np.diff(xpos)
print("spacing min/max:", dx.min(), dx.max())

# ----- MODEL -----

def model(params, x):
    A, w, phi, C, B, tau = params
    return A * np.sin(w * x + phi) + C + B * np.exp(-x / tau)

def ssr(params, x, y):
    residuals = y - model(params, x)
    return np.sum(residuals**2)

# ----- Starting guesses -----

w0 = 5
A0 = 7
C0 = -6
phi0 = 0.3
B0= -15
tau0 = 0.05
p0 = np.array([A0, w0, phi0, C0, B0, tau0])

# ----- Minimizer -----


def gradient_descent(f, p0, moving_rate=1e-6, max_runs=50000, tol=1e-10):
    p = np.array(p0, dtype=float)
    val = f(p)
    history = [(p.copy(), val)]

    for i in range(1, max_runs + 1):
        g = num_grad(f, p)               # from the Newton code
        p_new = p - moving_rate * g
        val = f(p_new)
        history.append((p_new.copy(), val))

        if not np.isfinite(val) or val > 1e20:   # blew up: rate too large
            return p_new, val, history, None

        moved = np.linalg.norm(p_new - p)
        p = p_new
        if moved < tol * (1 + np.linalg.norm(p)):  # stopped moving
            return p, val, history, i

    return p, val, history, None




def num_grad(f, p, h=1e-6):
    g = np.zeros(len(p))
    for i in range(len(p)):
        hi = h * max(1.0, abs(p[i]))          # scale step to parameter size
        e = np.zeros(len(p)); e[i] = hi
        g[i] = (f(p + e) - f(p - e)) / (2 * hi)
    return g

def num_hess(f, p, h=1e-4):
    n = len(p)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            hi = h * max(1.0, abs(p[i]))
            hj = h * max(1.0, abs(p[j]))
            ei = np.zeros(n); ei[i] = hi
            ej = np.zeros(n); ej[j] = hj
            H[i, j] = H[j, i] = (f(p + ei + ej) - f(p + ei - ej)
                                 - f(p - ei + ej) + f(p - ei - ej)) / (4 * hi * hj)
    return H

def newton(f, p0, max_runs=200, tol=1e-8):
    p = np.array(p0, dtype=float)
    val = f(p)
    history = [(p.copy(), val)]

    for i in range(1, max_runs + 1):
        g = num_grad(f, p)
        H = num_hess(f, p)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return p, val, history, None

        # if the Newton step points uphill, fall back to plain downhill
        if g @ step <= 0:
            step = g / (np.linalg.norm(g) + 1e-12)

        # backtrack: halve the step until the fit actually improves
        t = 1.0
        while t > 1e-10:
            p_new = p - t * step
            v_new = f(p_new)
            if np.isfinite(v_new) and v_new < val:
                break
            t /= 2
        else:
            return p, val, history, None          # no improving step found

        moved = np.linalg.norm(p_new - p)
        p, val = p_new, v_new
        history.append((p.copy(), val))

        if moved < tol * (1 + np.linalg.norm(p)): # stopped moving: converged
            return p, val, history, i

    return p, val, history, None




def nelder_mead(f, p0, scale=None, max_runs=5000, tol=1e-10):
    p0 = np.array(p0, dtype=float)
    n = len(p0)

    # per-parameter nudge: 10% of each parameter's size
    if scale is None:
        scale = 0.1 * np.maximum(np.abs(p0), 1e-2)

    # simplex has n+1 corners: the start, plus one nudged along each parameter
    simplex = np.vstack([p0] + [p0 + np.eye(n)[i] * scale[i] for i in range(n)])
    history = []

    for i in range(1, max_runs + 1):
        vals = np.array([f(v) for v in simplex])
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]

        best = simplex[0]
        history.append((best.copy(), vals[0]))

        # converged when the corners have bunched up AND their heights agree
        size = np.max(np.abs(simplex[1:] - best))
        spread = abs(vals[-1] - vals[0])
        if size < tol * (1 + np.linalg.norm(best)) and spread < tol * (1 + abs(vals[0])):
            return best, vals[0], history, i

        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]

        refl = centroid + (centroid - worst)
        f_refl = f(refl)

        if vals[0] <= f_refl < vals[-2]:             # second-worst, not [1], once n > 2
            simplex[-1] = refl
        elif f_refl < vals[0]:
            expand = centroid + 2 * (centroid - worst)
            simplex[-1] = expand if f(expand) < f_refl else refl
        else:
            contract = centroid + 0.5 * (worst - centroid)
            if f(contract) < vals[-1]:
                simplex[-1] = contract
            else:
                simplex[1:] = best + 0.5 * (simplex[1:] - best)

    return best, vals[0], history, None


# ----- Fit -----

objective = lambda p: ssr(p, xpos, ypos)
grad_best, final_ssr, hist, iters = gradient_descent(objective, p0)
new_best, final_ssr, hist, iters = newton(objective, p0)
neld_best, final_ssr, hist, iters = nelder_mead(objective, p0)

# ----- Plot -----
plt.scatter(xpos, ypos, s=8, label='data')
plt.plot(xpos, model(p0, xpos), 'r', label='initial guess')
# plt.plot(xpos, model(grad_best, xpos), 'g', label='Gradient Decent fit')
plt.plot(xpos, model(new_best, xpos), 'm', label='Newton fit', linewidth=4)
plt.plot(xpos, model(neld_best, xpos), 'k', label='Nelder Mead fit')
plt.legend()
plt.show()