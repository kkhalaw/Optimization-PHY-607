import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen
from numpy.linalg import inv

# ----- Rosenbrock Graph -----

# Create a grid of points in the x-y plane
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)

# Initialize an array to store Rosenbrock function values
Z = np.zeros_like(X)

# Calculate the Rosenbrock function for each point on the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosen([X[i, j], Y[i, j]])

# Create the plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, cmap='plasma')
plt.title('Rosenbrock Function (using scipy.optimize.rosen)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='f(x, y)')
plt.show()

#Equations
def func(p):
    x, y = p 
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 

Corners = [(-2, -1), (-2, 3), (2, -1), (2, 3)]  

# ----- X and Y range -----

Xrange = (-2, 2)
Yrange = (-1, 3)

# ----- Estimating Slope (Finite Differences) -----

def slope (p, h=0.000001):
    x, y = p
    dfdx = (func((x + h, y)) - func((x - h, y))) / (2*h)
    dfdy = (func((x, y + h)) - func((x, y - h))) / (2*h)
    return np.array([dfdx, dfdy])

# second derivative 

def secderv(p, h=0.000001):
    x, y = p
    d2fdx2 = (func((x + h, y)) - 2 * func((x, y)) + func((x - h, y))) / h**2
    d2fdy2 = (func((x, y + h)) - 2 * func((x, y)) + func((x, y - h))) / h**2
    return np.array([d2fdx2, d2fdy2])

# Hessian a 2x2 matrix holding the local curvature 

def hessian (p ,h=0.0001):
    p = np.array(p, dtype=float)     # so that p + step works for tuples too
    H = np.zeros((2,2)) # make the array aka the matrix 
    for i in range(2): # loop twice, i = 0 for x direction and i = 1 for y direction
        step = np.zeros(2)
        step[i] = h
        H[:, i] = (slope(p + step) - slope(p - step)) / (2 * h) # stores the move in x and y (":" for all rows, "i" is the column), first pass fills in x and second fills in y 
    return H

def inverse_hessian(H):
    inverse = np.linalg.inv(H)
    return inverse

# ----- Brute Force / Explicit Exploration of the Grid -----

def brute_force(n):
    xlist = np.linspace(-2, 2, n)
    ylist = np.linspace(-1, 3, n) 
    count = 0
    best = np.inf       # starter value for best (positive infinity)
    best_points = None  # value starts empty 
    for xval in xlist:
        for yval in ylist:
            count += 1 
            val = func(np.array([xval, yval]))
            if val < best:
                best = val                # replace best with the new best value 
                best_points = (xval, yval)
    return best_points, best, count 

best_points, best, count = brute_force(400)
print('best points:', best_points, ' best value:', best, ' count:', count)

# ----- Gradient Descent (gradient from finite differences) -----

def gradient_descent (start_x, start_y, moving_rate, max_runs=2000000, tol=1e-10):
    xg = start_x
    yg = start_y
    history = []

    # Gradient Descent Iterations 
    for i in range(1,max_runs+1):
        grad_x, grad_y = slope((xg, yg), 0.000001)

        xg = xg - moving_rate * grad_x
        yg = yg - moving_rate * grad_y
        val = func((xg, yg))

        history.append((xg, yg, val))

        if np.linalg.norm([grad_x, grad_y]) < tol:  # close enough: stop and report
             return xg, yg, val, history, i
        if not np.isfinite(val) or val > 1e20: # blew up: step size too large
            return xg, yg, val, history, None
    
    return xg, yg, val, history, None          # never got there in max_runs

# Test the four starting corners
for corner in Corners:
    x_opt, y_opt, f_opt, history, iterations = gradient_descent(corner[0], corner[1], moving_rate=1e-6)
    history = np.array(history)

    plt.figure(figsize=(8, 6))

    # Background: Rosenbrock heatmap (reuses X, Y, Z from the top of your script)
    plt.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')
    plt.colorbar(label='f(x, y)')

    # Overlay: gradient descent path
    plt.plot(history[:, 0], history[:, 1], '-o', color='white',
             markersize=2, linewidth=1, label='Descent path')

    # Mark the start and end points
    plt.plot(corner[0], corner[1], 'go', markersize=8, label='Start')
    plt.plot(x_opt, y_opt, 'r*', markersize=12, label='End')

    # Keep the view locked to the original grid range
    plt.xlim(Xrange)
    plt.ylim(Yrange)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Gradient Descent Path for {corner}")
    plt.legend()
    plt.show()

    print(
        f"Start: {corner}, "
        f"Final point: ({x_opt:.6f}, {y_opt:.6f}), "
        f"Function value: {f_opt:.6e}, "
        f"Iterations: {iterations}"
    )

# ----- Newton's Method -----

def newton(start_x, start_y, max_runs=1000, tol=1e-10):
    p = np.array([start_x, start_y], dtype=float)    # current position as an array
    history = [(p[0], p[1], func(p))]                # include the starting point

    for i in range(1, max_runs + 1):
        g = slope(p)                                 # which way is uphill?
        H = hessian(p)                               # how is the ground curving?

        try:
            step = np.linalg.solve(H, g)             # solve H * step = g
        except np.linalg.LinAlgError:                # H can't be "divided by"
            return p[0], p[1], func(p), history, None

        p = p - step                                 # jump to the bowl's bottom
        val = func(p)
        history.append((p[0], p[1], val))

        if val < tol:                                # close enough: stop and report
            return p[0], p[1], val, history, i
        if not np.isfinite(val):                     # numbers blew up
            return p[0], p[1], val, history, None

    return p[0], p[1], val, history, None            # never got there

print(newton(-2, -1)[-1])     # iteration count; should be about 6
print(newton(-2, -1)[:3])     # end x, end y, end value

# ----- Plot the path a method takes -----

def plot_path(history, title):
    path = np.array(history)              # each row is (x, y, f-value)
    px, py = path[:, 0], path[:, 1]       # all x's, all y's along the path

    plt.figure(figsize=(8, 6))

    # Background map. log10 squashes the huge range of values so the narrow
    # valley is visible. The 1e-3 stops log10 from choking on a value of 0.
    plt.contourf(X, Y, np.log10(Z + 1e-3), levels=40, cmap='plasma')
    plt.colorbar(label='log10( f(x, y) )')

    plt.plot(px, py, 'w.-', linewidth=1.5, markersize=8, label='path')
    plt.plot(px[0], py[0], 'go', markersize=10, label='start')
    plt.plot(1, 1, 'r*', markersize=16, label='true minimum (1, 1)')

    plt.xlim(-2, 2)
    plt.ylim(-1, 3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()

# ----- Run Newton's method from each corner and plot it -----

for corner in Corners:
    xn, yn, val, history, iters = newton(corner[0], corner[1])
    print('start', corner, '-> ended at', (xn, yn), ' iterations:', iters)
    plot_path(history, f"Newton's method, start {corner} ({iters} iterations)")


plt.figure(figsize=(8, 6))
plt.contourf(X, Y, np.log10(Z + 1e-3), levels=40, cmap='plasma')
colors = ['white', 'cyan', 'lime', 'yellow']
for corner, c in zip(Corners, colors):
    _, _, _, history, iters = newton(corner[0], corner[1])
    path = np.array(history)
    plt.plot(path[:, 0], path[:, 1], '.-', color=c, label=f'start {corner}')
plt.plot(1, 1, 'r*', markersize=16)
plt.xlim(-2, 2); plt.ylim(-1, 3)
plt.legend(); plt.show()

# ----- Nelder-Mead ----- (Tuesday 10/20)

