import numpy as np

import h5py
from matplotlib import pyplot as plt

f = h5py.File('./data.hdf', 'r')

print(f.keys())

print(f['data'].keys())

xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]



def fit(x, a, b, c, d, e, f, g, w):

    result = a + b * np.sin(w*x) + c * np.cos(w*x) + d * np.sin(2*w*x) + e * np.cos(2*w*x) + f * np.sin(3*w*x) + g * np.cos(3*w*x)
    
    return result


def doublemulti(a, b, c, d, e, f, g, w):

    return np.sum((ypos - fit(xpos, a, b, c, d, e, f, g, w))**2)


p = np.array([np.mean(ypos), np.std(ypos), 0, 0, 0, 0, 0, 2*np.pi/(max(xpos)-min(xpos))])

h = 0.1
scale = np.array([1, 1, 1, 1, 1, 1, 1, p[7]])
fbest = doublemulti(*p)

count = 0

while h > 1e-6 and count < 10000:

    count += 1

    points = []

    for i in range(8):
        for s in [-1, 1]:
            q = p.copy()
            q[i] += s*h*scale[i]
            points.append(q)

    values = [doublemulti(*q) for q in points]

    Potentialmin = min(values)
    arg = np.argmin(values)

    if Potentialmin < fbest:
        fbest = Potentialmin
        p = points[arg]

    else:
        h = h/2

print(p, fbest)

x = np.linspace(min(xpos), max(xpos), 500)
y = fit(x, *p)

plt.scatter(xpos, ypos)
plt.plot(x, y, "w")
plt.show()