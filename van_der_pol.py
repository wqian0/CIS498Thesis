from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.ndimage import convolve
from skimage.filters import gaussian
from sklearn.neighbors import KernelDensity
import pickle as pk
from sklearn.utils.extmath import cartesian
def vdp(t, z, mu = 0):
    x, y = z
    return [y, mu * (1 - x ** 2) * y - x]

# def vdp_learned(t, z, mu = 0):
#     x, y = z
#     return [y*0.5, (mu * (1 - x ** 2) * y - x)*0.5]
def vdp_learned(t, z, mu = 0):
    x, y = z
    return [y*0.3826, y* 0.6608 + x**2 * y *(-.1431)]

# def vdp(t, z, mu = 0):
#     x, y = z
#     return [y, mu* (1 - x ** 2) * y - x]

def points_on_circle(r,n=100, offset = None):
    if offset is not None:
        x_off,y_off = offset
        return [(np.cos(2*np.pi/(n-1)*x)*r - x_off,np.sin(2*np.pi/(n-1)*x)*r-y_off) for x in range(n)]
    return [(np.cos(2*np.pi/(n-1)*x)*r,np.sin(2*np.pi/(n-1)*x)*r) for x in range(n)]

def compute_densities(data, x_range, y_range, num_x, num_y, t):
    T = len(t)
    output = []
    for i in range(T):
        H, _, _ = np.histogram2d(data[:,0, i], data[:, 1, i], bins = [num_x, num_y], range = [x_range, y_range], density = True)
        output.append(H)
    return output

def compute_densities_KDE(data, x_range, y_range, num_x, num_y, t, bandwidth = 0.1):
    T = len(t)
    output = []
    xx, yy = np.mgrid[x_range[0]:x_range[1]:num_x*1j, y_range[0]:y_range[1]:num_y*1j]
    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T
    for i in range(T):
        kde = KernelDensity(kernel = 'gaussian', bandwidth=bandwidth, atol = 1e-9, rtol = 1e-9)
        kde.fit(data[:,:,i])
        z = np.exp(kde.score_samples(xy_sample))
        output.append(np.reshape(z, xx.shape))
    return output

def van_der_pol(x_range = [-6,6], y_range = [-6,6], num_x = 100, num_y = 100, t_max=10, dt= 0.05, mu = 2, n = 5000, r = 3, noise = 0.0, offset = None, bandwidth = 0, ic_mode = 'circle', vdp_func = None):
    t = np.linspace(0, t_max, int(t_max  / dt) + 1)
    if ic_mode == 'circle':
        init_conditions = points_on_circle(r, n=n, offset=offset)
    elif ic_mode == 'uniform':
        init_conditions = np.random.random((n,2)) * (x_range[1] - x_range[0]) - x_range[1]
    else:
        x = np.linspace(x_range[0], x_range[1], num_x)
        y = np.linspace(y_range[0], y_range[1], num_y)
        init_conditions = cartesian([x,y])
        n = len(init_conditions)
    all_data = np.zeros((n, 2, len(t)))
    if vdp_func is None:
        vdp_func = vdp
    for i in range(len(init_conditions)):
        if i % 10000 == 0:
            print(i)
        x, y = init_conditions[i][0], init_conditions[i][1]
        sol = solve_ivp(lambda t, z: vdp_func(t, z, mu=mu), [t[0], t[-1]], [x, y], t_eval=t)
        all_data[i] = sol.y
        if noise > 0:
            all_data[i] += np.random.normal(scale = noise, size = sol.y.shape)
    if bandwidth > 0:
        H_list = compute_densities_KDE(all_data,x_range,y_range,num_x,num_y,t,bandwidth=bandwidth)
    else:
        H_list = compute_densities(all_data, x_range, y_range, num_x, num_y, t)
    H_list = np.array(H_list)
    return H_list
if __name__ == "__main__":
    # a, b = 0, 20
    #
    # mus = [0, 1, 2]
    # styles = ["-", "--", ":"]
    # t = np.linspace(a, b, 500)
    # plt.figure(dpi = 200)
    # for mu, style in zip(mus, styles):
    #     sol = solve_ivp(lambda t,z :vdp(t,z,mu = mu), [a, b], [1, 0], t_eval=t)
    #     plt.plot(sol.y[0][200:], sol.y[1][200:], style)
    #
    # # make a little extra horizontal room for legend
    # plt.xlim([-3, 3])
    # plt.ylim([-5, 5])
    # plt.legend([f"$\mu={m}$" for m in mus])
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$y$')
    # ax = plt.gca()  # you first need to get the axis handle
    # ax.set_aspect(1)
    #plt.axes().set_aspect(1)
    # plt.show()
    n = 10000
    # #H_list = van_der_pol(x_range = [-6,6], y_range = [-6,6], num_x = 256, num_y = 256, n = n, dt = 0.005, t_max=1, noise = 0, bandwidth=0, ic_mode='circle')
    H_list = van_der_pol(x_range=[-6, 6], y_range=[-6, 6], num_x=512, num_y=512, n=n, dt=0.05, t_max=10, noise=0,
                          bandwidth=0, ic_mode='circle', vdp_func=vdp_learned)
    # #H_list = pk.load(open("50k_256_vdp.pk", "rb"))
    # H_list = pk.load(open("already_generated_300k.pk", "rb"))['u']
    # H_list = np.transpose(H_list, (2,0,1))
    # H_list_orig = deepcopy(H_list)
    # print(H_list.shape)
    # p = np.zeros((256, 256))
    # p[[127, 128], [127, 128]] = 1 / 2
    # p = gaussian(p, sigma=200)
    # p /= np.sum(p)
    #
    # plt.imshow(p)
    # plt.show()
    # idxes = cartesian((np.arange(512), np.arange(512)))
    # sub_xyz = np.sort(np.random.choice(len(idxes), size=int(len(idxes)*0.03), p=p.flatten(), replace=False))
    # print(idxes[sub_xyz])
    # H_list = np.reshape(H_list, (201, -1))
    # H_list = H_list[:, sub_xyz]
    # print(H_list.shape)
    # print(np.sum(H_list))

    #pk.dump(H_list, open("50k_256_vdp.pk","wb"))
    print(np.count_nonzero(H_list)/ H_list.size)
    print(H_list.shape)
    #print(np.amax(H_list), np.amax(H_list_fake),n)
    #plt.figure()
    for i in range(len(H_list)):
        plt.imshow(H_list[i].T, vmin = 0, vmax = np.amax(H_list)/20, cmap = 'hot', origin = 'lower')
        #print(np.sum(H_list[i]))
        plt.pause(0.01)
        plt.clf()
    # fig, axes = plt.subplots(2, 3)
    # #for ax in axes.flat:
    #     #ax.set(xlabel=r'$x$', ylabel=r'$y$')
    # for j in range(6):
    #     for i in range(len(H_list)):
    #         H_list[i] = gaussian(H_list_orig[i], sigma=j)
    #         # H_list_fake[i] = gaussian(H_list_fake[i], sigma = 0)
    #     #plt.figure(dpi = 200)
    #     axes[j // 3, j % 3].set_title('Coarsening Level ' + str(j + 1))
    #     axes[j // 3, j % 3].imshow(H_list[len(H_list) // 2].T, vmax = np.amax(H_list_orig)/100, cmap = 'hot', origin = 'lower')
    #     axes[j // 3, j % 3].axis('off')
    # plt.tight_layout()
    # plt.savefig('vdp_coarsening.png', dpi = 200)
    plt.show()