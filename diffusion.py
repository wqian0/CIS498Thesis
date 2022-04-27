import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
import random
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from scipy.linalg import eigh
from copy import copy, deepcopy
import pickle as pk
import random
import scipy
from scipy.integrate import quad

def discrete_ft(theta, k, N=3):
    return np.sum([np.exp(-1j * k * u)*theta[u + N] for u in range(-N, N+1)])

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return func(x).real
    def imag_func(x):
        return func(x).imag
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


def compute_metric(theta, t, N):
    prefactor = t ** 2 / (2 * np.pi)
    n_params = len(theta)
    g = np.zeros((n_params, n_params))
    for u in range(n_params):
        for v in range(u, n_params):
            g[u][v] = complex_quadrature(lambda k: prefactor * np.exp(1j * k * (u - v)) * discrete_ft(theta, k, N = N) ** (t-1) * discrete_ft(theta, -k, N = N) ** (t-1), -np.pi, np.pi)[0].real
            g[v][u] = g[u][v]
    w, v = eigh(g)
    return g, w

def empirical_sim(balls_loc_init, theta, steps, trials, donations = None):
    N = (len(theta) - 1) // 2
    N_balls = len(balls_loc_init)
    tot_counts = defaultdict(int)
    for i in range(trials):
        balls_loc = deepcopy(balls_loc_init)
        for j in range(steps):
            changes = choice(np.arange(-N, N + 1), len(balls_loc), p=theta, replace = True)
            balls_loc_prev = deepcopy(balls_loc)
            balls_loc += changes
            if donations is not None:
                matches = changes == (donations[0] - donations[2])
                num_matches = np.count_nonzero(matches)
                conditions = np.random.rand(num_matches) < donations[1]/theta[donations[0]]
                donated = balls_loc_prev[matches][conditions]
                exclude = np.array([donations[0] - donations[2]])
                donated += choice(np.setdiff1d(np.arange(-N, N + 1), exclude), len(donated), replace = True)
                balls_loc = np.concatenate([balls_loc, donated])
        vals, counts = np.unique(balls_loc, return_counts=True)
        for i in range(len(vals)):
            tot_counts[vals[i]] += counts[i]
    for loc in tot_counts:
        tot_counts[loc] /= trials * N_balls #fractional estimate
    return tot_counts

def compute_empirical_jacobian(balls_loc_init, theta, steps, trials, meta_trials, pred_max, delta = .001):
    theta_0 = deepcopy(theta)
    fracs_0 = empirical_sim(balls_loc_init, theta_0, steps, trials)
    n_params = len(theta)
    J = np.zeros((pred_max * 2 + 1, n_params))
    for i in range(meta_trials):
        print(i)
        for j in range(len(theta)):
            shift = np.zeros(len(theta))
            shift[j] += delta * (n_params) / (n_params - 1)
            shift -= delta * 1 / (n_params - 1)
            theta_cur = shift + theta_0
            fracs = empirical_sim(balls_loc_init, theta_cur, steps, trials, donations = [j, delta, N])
            for r in range(len(J)):
                shifted_r = r - pred_max
                J[r][j] += (fracs[shifted_r] - fracs_0[shifted_r]) / delta
        #print(J / (i+1))
    J /= meta_trials
    g = J.T @ J
    w, v = eigh(g)
    return g, w

e_vals = []
N = 10
theta = np.ones(2 * N + 1) / (2 * N + 1)
theta = np.random.rand(2 * N + 1)
theta /= theta.sum()

balls_loc_init = np.zeros(20000)

# fracs = empirical_sim(balls_loc_init, theta, 1, 1000)
# print(fracs)

# g, w = compute_empirical_jacobian(balls_loc_init, theta, 1, 10000, 2, N, delta = .01)
# w = np.sort(w)[::-1]
# print(w)
# print(g)
for t in range(1, 7):
    g, w = compute_metric(theta, t, N)
    print("time elapsed", t)
    #g, w = compute_empirical_jacobian(balls_loc_init, theta, t, 10000, 1, t * N, delta=.01)
    w = np.sort(w)[::-1]
    e_vals.append(w)
e_vals = [np.sort(w)[::-1] for w in e_vals]
plt.figure()
ax = plt.gca()
ax.set_yscale('log')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Coarsening level (elapsed time)')
plt.ylabel('Eigenvalue')
#plt.ylim([1e-3, 1e8])
for i in range(len(e_vals)):
    plt.scatter(np.ones_like(e_vals[i]) * (i + 1), e_vals[i], marker="_", s= 150)
plt.show()