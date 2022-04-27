import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
from scipy.ndimage import convolve
import tqdm
from scipy.linalg import eigh
from copy import copy, deepcopy
import pickle as pk
import random
import pde
import sys
from skimage.filters import gaussian
from pde import CartesianGrid, DiffusionPDE, ScalarField, plot_kymograph, MemoryStorage
import van_der_pol as vdp
import matplotlib.colors as colors


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def get_eqn_lib():
    const = '1'
    u = 'φ'
    ux = 'get_x(gradient((φ)))'
    uxx = 'laplace(φ)'
    uxxx = 'laplace(get_x(gradient((φ))))'
    multiply = lambda a, b: str(a) + '*' + str(b)
    raise_pow = lambda a, b: str(a) + '**' + str(b)

    u_sq = raise_pow(u, 2)
    u_cb = raise_pow(u, 3)

    lib = [const, u, u_sq, u_cb, ux, multiply(u, ux), multiply(u_sq, ux), multiply(u_cb, ux), uxx, multiply(u, ux),
           multiply(u_sq, ux), multiply(u_cb, ux), uxxx, multiply(u, uxxx), multiply(u_sq, uxxx), multiply(u_cb, uxxx)]
    return lib

def form_eqn(coeffs):
    user_funcs = {"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1]}
    add = lambda a, b: str(a) + '+' + str(b)
    lib = get_eqn_lib()
    eqn =''
    for i in range(len(coeffs)):
        if coeffs[i] != 0:
            eqn = add(eqn, str(coeffs[i])+'*'+lib[i])
    return pde.PDE({"φ": eqn}, user_funcs=user_funcs)

def solve_2d_advection(x_range,num_x, y_range, num_y, t_max, dt, trackers = 'auto', mu = 2, init_cond = None):
    user_funcs = {"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1], "mu": mu}
    eq_string = '- get_x(gradient(φ))*y - get_y(gradient(φ))*('+str(mu)+'*(1-x*x)*y - x)-φ*'+str(mu)+'*(1-x*x)'
    eq = pde.PDE({"φ": eq_string}, user_funcs=user_funcs)
    grid = CartesianGrid([[x_range[0], x_range[1]], [y_range[0], y_range[1]]], [num_x, num_y], periodic=True)
    state = ScalarField(grid)
    if init_cond is not None:
        state.data = init_cond
    else:
        state.data = np.random.normal(scale = 0.5)
    x = np.linspace(x_range[0], x_range[1], num_x)
    y = np.linspace(y_range[0], y_range[1], num_y)
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    storage = pde.MemoryStorage()
    trackers = [
        "progress",  # show progress bar during simulation
        storage.tracker(dt),  # store data every simulation time unit
        # print some output every 5 real seconds:
    ]
    try:
        result = eq.solve(state, t_range=t_max, tracker=trackers, rtol=1e-15, atol=1e-15)
    except ValueError as e:
        result = None
        print(e)
    return result, np.array(storage.data), x,y, t

def solve_rand_eqn(x_min, x_max, num_x, t_max, dt, trackers = 'auto', num_coeff = 16, num_terms  = 2, init_cond = None):
    coeffs = np.zeros(16)
    # coeffs[5] = -.8837
    # coeffs[8] = 0.094
    coeffs[4] = -3.5964
    coeffs[5] = -.1708
    coeffs[6] = -0.2476
    coeffs[-4] = -0.1388
    coeffs[-3] = -0.06313

    print(coeffs)
    eq = form_eqn(coeffs)
    grid = CartesianGrid([[x_min, x_max]], [num_x], periodic=False)
    state = ScalarField(grid)
    if init_cond is not None:
        state.data = init_cond
    else:
        state.data = np.random.normal(scale = 0.5)
    x = np.linspace(x_min, x_max, num_x)
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    storage = pde.MemoryStorage()
    trackers = [
        "progress",  # show progress bar during simulation
        storage.tracker(dt),  # store data every simulation time unit
        # print some output every 5 real seconds:
    ]
    try:
        result = eq.solve(state, t_range=t_max, tracker=trackers, rtol=1e-15, atol=1e-15)
    except ValueError as e:
        result = None
        print(e)
    return result, np.array(storage.data), x, t

def burgers_eq(x_min = -8, x_max = 8, num_x = 256, t_max = 10, dt = 0.1, perturbation = '', trackers = 'auto', init_cond = None):
    user_funcs = {"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1]}
    eq_string = '- φ * get_x(gradient((φ))) + 0.1*laplace(φ)' + perturbation
    eq = pde.PDE({"φ": eq_string}, user_funcs=user_funcs)
    grid = CartesianGrid([[x_min, x_max]], [num_x], periodic=False)
    # state = ScalarField.from_expression(grid, "sin(x)")
    state = ScalarField(grid)
    x = np.linspace(x_min, x_max, num_x)
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    if init_cond is not None:
        state.data = init_cond
    # solve the equation and store the trajectory
    storage = pde.MemoryStorage()
    trackers = [
        "progress",  # show progress bar during simulation
        storage.tracker(dt),  # store data every simulation time unit
        # print some output every 5 real seconds:
    ]
    pde.registered_solvers()
    result = eq.solve(state, t_range=t_max, tracker=trackers, rtol=1e-15, atol=1e-15)
    return result, np.array(storage.data), x, t

def soliton_periodic(x,t,c,a = 0, x_min=-30, x_max = 30):
    arg = x-c*t
    while np.any(np.bitwise_or(arg<x_min, arg >x_max)):
        arg[arg<x_min] += x_max - x_min
        arg[arg >x_max] -= x_max - x_min
    return c/2*np.cosh(np.sqrt(c)/2*(arg-a))**-2

def soliton(x,t,c,a = 0):
    return c/2*np.cosh(np.sqrt(c)/2*(x-c*t-a))**-2
def get_true_solution_kdv(solitons, x, t, periodic = False):
    ans = 0
    for s in solitons:
        if periodic:
            ans += soliton_periodic(x[:, None], t, s[1], a=s[0], x_min = x[0], x_max = x[-1])
        else:
            ans += soliton(x[:, None], t, s[1], a = s[0])
    return ans
def kortweg_de_vries_equation(origin_1, c1, origin_2, c2, x_min=-30, x_max=30, num_x=512, t_max=20, dt = 0.01, perturbation = "", trackers = "auto"):
    user_funcs = {"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1]}
    eq_string = '-6 * φ * get_x(gradient((φ))) - laplace(get_x(gradient((φ))))'+perturbation
    eq = pde.PDE({"φ": eq_string}, user_funcs=user_funcs)
    grid = CartesianGrid([[x_min, x_max]], [num_x], periodic=True)
    #state = ScalarField.from_expression(grid, "sin(x)")
    state = ScalarField(grid)
    x = np.linspace(x_min, x_max, num_x)
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    state.data = soliton(x, 0, c1, a = origin_1) + soliton(x-origin_2, 0, c2, a = origin_2)
    # solve the equation and store the trajectory
    storage = pde.MemoryStorage()
    trackers = [
        "progress",  # show progress bar during simulation
        storage.tracker(dt),  # store data every simulation time unit
        # print some output every 5 real seconds:
    ]
    result = eq.solve(state, t_range = t_max, tracker=trackers, rtol = 1e-12, atol = 1e-12)
    return result, np.array(storage.data), x, t


def kortweg_de_vries(solitons, init_cond = None, x_min=-30, x_max=30, num_x=512, t_max=20, dt = 0.01, perturbation = "", trackers = "auto", periodic = False):
    user_funcs = {"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1]}
    eq_string = '-6 * φ * get_x(gradient((φ))) - laplace(get_x(gradient((φ))))'+perturbation
    eq = pde.PDE({"φ": eq_string}, user_funcs=user_funcs)
    grid = CartesianGrid([[x_min, x_max]], [num_x], periodic=periodic)
    #state = ScalarField.from_expression(grid, "sin(x)")
    state = ScalarField(grid)
    x = np.linspace(x_min, x_max, num_x)
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    if init_cond is not None:
        state.data = init_cond
    else:
        for s in solitons:
            state.data += soliton(x, 0, s[1], a = s[0])
    # solve the equation and store the trajectory
    storage = pde.MemoryStorage()
    trackers = [
        "progress",  # show progress bar during simulation
        storage.tracker(dt),  # store data every simulation time unit
        # print some output every 5 real seconds:
    ]
    pde.registered_solvers()
    result = eq.solve(state, t_range = t_max, tracker=trackers, rtol = 1e-15, atol = 1e-15)
    #result = eq.solve(state, t_range=t_max, tracker=trackers)
    return result, np.array(storage.data), x, t

def one_d_equation_example(trackers = "auto"):
    user_funcs = {"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1], "squared_sum": lambda arr: np.sum(np.square(arr))}
    eq = pde.PDE({"u": "-0.5*squared_sum(get_x(gradient(u))) + laplace(u)"}, user_funcs = user_funcs)
    grid = CartesianGrid([[0, 2 * np.pi]], 100, periodic=True)
    state = ScalarField.from_expression(grid, "sin(x)")
    # solve the equation and store the trajectory
    storage = MemoryStorage()
    result = eq.solve(state, t_range=3, tracker=storage.tracker(0.01))
    plot_kymograph(storage)
    return result

def diffusion_equation(x_max, dx, t_max, dt, diffusivity = 0.1, trackers = "auto"):
    grid = CartesianGrid([[0, x_max], [0, x_max]], [x_max//dx, x_max//dx])  # generate grid
    state = ScalarField(grid)  # generate initial condition
    state.insert([x_max // 4, x_max//4], 1)
    state.insert([3 * x_max // 4, 3 * x_max // 4], 1)

    #eq = DiffusionPDE(diffusivity=0.1)  # define the pde
    eq = pde.PDE({"u": "laplace(u)/10 - get_y(gradient(get_x(gradient(u))))/10"}, user_funcs={"get_x": lambda arr: arr[0], "get_y": lambda arr: arr[1]})
    result = eq.solve(state, t_range=t_max, dt=dt, tracker = trackers)
    return result
if __name__ == "__main__":
    storage = pde.MemoryStorage()
    trackers = [
        "progress",  # show progress bar during simulation
        storage.tracker(interval=1),  # store data every simulation time unit
        # print some output every 5 real seconds:
        pde.PrintTracker(interval=pde.RealtimeIntervals(duration=5)),
    ]
    # H_list = vdp.van_der_pol(x_range = [-6,6], y_range = [-6,6], num_x = 256, num_y = 256, n = 50000, dt = 0.05, noise = 0, bandwidth=0.1)
    # weights = np.zeros((5, 5))
    # weights[2, 2] = 1
    # weights = gaussian(weights, sigma=1)
    # H_list = np.transpose(H_list, (1,2,0))
    # plt.figure()
    # plt.imshow(H_list[:,:,0], cmap = 'hot')
    # plt.show()
    # result, data, x,y,t = solve_2d_advection([-6, 6], 256, [-6,6], 256, 1, 0.06, init_cond=H_list[:,:,0])
    # print(data.shape)
    # plt.figure()
    # plt.imshow(data[0], cmap = 'hot')
    # plt.figure()
    # plt.imshow(data[1], cmap = 'hot')
    # plt.figure()
    # plt.imshow(data[2], cmap = 'hot')

    # plt.figure()
    # for i in range(len(data)):
    #     plt.imshow(data[i], vmin=0, vmax=np.amax(H_list) / 10, cmap='hot')
    #     print(np.sum(data[i]))
    #     plt.pause(0.1)
    #     plt.clf()
    # plt.imshow(data[-1], vmin=0, vmax=np.amax(H_list) / 10, cmap='hot')
    # plt.show()



    f = scipy.io.loadmat('burgers.mat')
    u = np.real(f['usol'])
    u, x, t = np.real(f['usol']), f['x'], f['t']
    x = x.flatten()
    t = t.flatten()
    # plt.figure()
    # plt.imshow(u, cmap = 'hot', vmax = np.max(u))
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$x$')
    # plt.colorbar()
    x = np.linspace(-16,16,512)
    t = np.linspace(0,5, int(5/0.02)+1)
    init = soliton(x, 0, 4, a = -10) + soliton(x,0,1,a=0)
    result, data , _, _ = solve_rand_eqn(init_cond = init,x_min = x[0], x_max = x[-1], t_max = t[-1], dt = t[-1] / (len(t) - 1), num_x = len(x))
    result, u, _, _ = kortweg_de_vries([(-10, 4), (0, 1)], x_min = x[0], x_max=x[-1], t_max = t[-1], dt= t[-1] / (len(t) - 1), num_x = len(x))
    # plt.figure()
    # plt.imshow(data.T, cmap = 'hot', vmax = np.max(u))
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$x$')
    # plt.colorbar()

    norm = MidpointNormalize(midpoint=0, vmin=-np.max(u) / 2, vmax=np.max(u) / 2)
    # plt.figure()
    # plt.imshow(u - data.T, cmap = 'bwr', norm = norm)
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$x$')
    # plt.colorbar()


    fig, axes = plt.subplots(1,3)
    for ax in axes.flat:
        ax.set(xlabel=r'$t$', ylabel=r'$x$')
    axes[0].set_title(r'$u$')
    im0 = axes[0].imshow(u.T, cmap = 'hot', vmax = np.max(u), aspect = 0.6)
    plt.colorbar(im0, ax=axes[0], fraction=0.056, pad=0.05)
    axes[1].set_title(r'$\hat{u}$')
    im1 = axes[1].imshow(data.T, cmap='hot', vmin = 0, vmax=np.max(u), aspect = 0.6)
    plt.colorbar(im1, ax=axes[1], fraction=0.056, pad=0.05)
    axes[2].set_title(r'$u - \hat{u}$')
    im2 = axes[2].imshow(u.T - data.T, cmap = 'bwr', norm = norm, aspect = 0.6)
    plt.colorbar(im2, ax=axes[2], fraction=0.056, pad=0.05)
    plt.tight_layout()
    plt.show()
    #
    # # x = np.linspace(-16, 16, 512)
    # # t = np.linspace(0, 5, 201)
    # # result, data, _, _ =  solve_rand_eqn(init_cond =soliton(x, 0, 4, a = -10) + soliton(x, 0, 1, a = 0), x_min = x[0], x_max = x[-1], t_max = t[-1], dt = t[-1] / (len(t) - 1), num_x = len(x))
    # # data = data.T
    # # result = diffusion_equation(16, 0.1, 10, 0.01, trackers=trackers)
    #
    # #result.plot(cmap="magma")
    #
    # # result, data, _, _ = kortweg_de_vries([(-2, 3), (2, 5)])
    # # result.plot()
    #
    # x = np.linspace(-16, 16, 256)
    # t = np.linspace(0, 5, 101)
    #
    #
    # # f = scipy.io.loadmat('kdv.mat')
    # # u, x, t = np.real(f['usol']), f['x'], f['t']
    #
    # perturbation = ''
    # result, data2, _, _ = kortweg_de_vries([(-10, 4), (0, 1)], x_min =x[0], x_max = x[-1], num_x = len(x), t_max = t[-1], dt = t[-1] / (len(t) - 1), perturbation='', periodic = False)
    # u = data2.T
    # # pde.registered_solvers()
    # # true_sol = get_true_solution_kdv([(-12, 2), (-9, 1)], x,
    # # t, periodic = True)
    #
    # # plt.figure()
    # # plt.imshow(u)
    # # plt.show()
    # # perturbation = '+.24266*φ * get_x(gradient((φ))) -0.063813*get_x(gradient((φ))) -0.018581*laplace(φ)' #fake burgers equation
    # # #perturbation = '+4.4158* φ * get_x(gradient((φ))) +.5417 * laplace(get_x(gradient((φ)))) '
    # # perturbation = '+0.5*laplace(φ)'
    # # # noise = np.random.normal(scale = 1, size = u[:,0].shape)
    # # result,data, x, t = burgers_eq(init_cond=u[:,0],t_max = 10, dt = 0.1, num_x = 256, perturbation='')
    # # data = data.T
    # #
    # # result,data2, x, t = burgers_eq(init_cond=u[:,0]+ noise,t_max = 10, dt = 0.1, num_x = 256, perturbation='')
    # # u = data2.T
    #
    # for i in range(0,len(t),1):
    #     plt.ylim([0,5])
    #     #plt.plot(x, data[:, i])
    #     #plt.plot(x, true_sol[:, i])
    #     plt.plot(x, u[:, i])
    #     plt.pause(0.1)
    #     plt.clf()
    # plt.ylim([0, 5])
    # #plt.plot(x,data[:,-1])
    #
    # # result, data, _, _ = kortweg_de_vries([(-2, 3), (2, 5)])
    # # result.plot()
    #
    # plt.show()