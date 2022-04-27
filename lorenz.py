from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from skimage.filters import gaussian
from sklearn.utils.extmath import cartesian
from sklearn.neighbors import KernelDensity
import pickle as pk
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import plotly.io as pio
def lorenz_dynamics(t, inp, rho = 28, sigma = 10, beta = 8/3):
    x, y, z = inp
    return [sigma * (y-x), x*(rho - z) - y, x*y - beta * z]

def lorenz_dynamics_rescaled(t, inp, rho = 28, sigma = 10, beta = 8/3, scale = 10):
    x, y, z = inp
    return np.array([sigma * (y-x), x*(rho - scale * z) - y, scale*x*y - beta * z])

def compute_densities(data, x_range, y_range, z_range, num_x, num_y, num_z, t):
    T = len(t)
    output = []
    for i in range(T):
        H, _ = np.histogramdd(data[:, :, i], bins = [num_x, num_y, num_z], range = [x_range, y_range, z_range], density = True)
        output.append(H)
    return output

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def lorenz(x_range = [-3,3], y_range = [-3,3], z_range = [0,5], num_x = 100, num_y = 100, num_z = 100, t_max=10, dt= 0.05, n = 5000, spherical_r = 0.0, return_raw = False):
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    if spherical_r == 0:
        init_x = np.random.random((n, 1)) * (x_range[1] - x_range[0]) + x_range[0]
        init_y = np.random.random((n, 1)) * (y_range[1] - y_range[0]) + y_range[0]
        init_z = np.random.random((n, 1)) * (z_range[1] - z_range[0]) + z_range[0]
        init_conditions = np.hstack([init_x, init_y, init_z])
    else:
        init_conditions = np.array([[np.mean(x_range), np.mean(y_range), np.mean(z_range)]]) + sample_spherical(n).T * spherical_r
    all_data = np.zeros((n, 3, len(t)))
    for i in range(len(init_conditions)):
        if i % 10000 == 0:
            print(i)
        x, y, z = init_conditions[i][0], init_conditions[i][1], init_conditions[i][2]
        sol = solve_ivp(lorenz_dynamics_rescaled, [t[0], t[-1]], [x, y, z], t_eval=t)
        all_data[i] = sol.y
    H_list = compute_densities(all_data, x_range, y_range,z_range, num_x, num_y,num_z, t)
    H_list = np.array(H_list, dtype = np.float32)
    if return_raw:
        return H_list, all_data
    return H_list
def plot3d(density):
    X, Y, Z = np.mgrid[-3:3:100j, -3:3:100j, 0:5:100j]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=density.flatten(),
        opacity=0.06,
        isomin=1e-3,
        isomax=np.max(density),
        surface_count=200,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    #fig.write_image("density.svg")
    fig.show()

def animate(raw_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c='blue', alpha=0.1, s = 20)

    def update(i):
        sc._offsets3d = (raw_data[:, 0, i], raw_data[:, 1, i], raw_data[:, 2, i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 5)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(raw_data), interval=70)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    n = 1000
    #H_list, raw_data = lorenz(dt = 0.01, n = n, t_max=6, num_x=100, num_y=100, num_z=100, spherical_r=2.0, return_raw=True)
    # H_list = pk.load(open("already_generated_300k_lorenz_spherical.pk","rb"))['u']
    # pio.renderers.default = 'browser'
    # print(H_list.shape)
    # p = np.zeros((100,100, 100))
    # p[[49,50], [49,50], [49,50]] = 1/2
    # p = gaussian(p, sigma = 20)
    # plt.imshow(p[49, :, :])
    # plt.show()
    # print(np.sum(p))
    # # sub_x = np.sort(np.random.choice(100, size = 10, p = p, replace = False))
    # # sub_y = np.sort(np.random.choice(100, size = 10, p = p, replace=  False))
    # # sub_z = np.sort(np.random.choice(100, size=10, p = p, replace = False))
    #idxes = cartesian((np.arange(100), np.arange(100)))
    # print(cartesian([idxes,np.arange(2)]).shape)
    #sub_xyz = np.sort(np.random.choice(len(idxes), size = 800, p = p.flatten(), replace=  False))
    # coords = idxes[sub_xyz]
    # print(coords)
    # print(coords.shape)
    # print(H_list.shape)
    # H_list = np.transpose(H_list, (3,0,1,2))
    # #plot3d(H_list[500])
    # for i in range(len(H_list)):
    #     H_list[i] = gaussian(H_list[i], sigma = 2)
    # plot3d(H_list[500])
    #
    # H_list = np.reshape(H_list, (1001, -1))
    # H_list = H_list[:, sub_xyz]
    # H_list = np.reshape(H_list, (1001, 20, 20, 20))
    # #H_list = H_list[:, sub_x, :, :]
    # #H_list = H_list[:, :, sub_y, :]
    # #H_list = H_list[:, :, :, sub_z]
    # print(H_list.shape)
    # print(np.amax(H_list[0]))
    # print(np.amax(H_list))
    # # for i in range(len(H_list)):
    # #     H_list[i] = gaussian(H_list[i], sigma = 2)
    # print(np.count_nonzero(H_list) / H_list.size)
    # for i in range(len(H_list)):
    #     plt.imshow(gaussian(H_list[i, :, :, 8], sigma = 0), vmin=0, vmax=np.amax(H_list) / 10, cmap='hot')
    #     # print(np.sum(H_list[i]))
    #     plt.pause(0.01)
    #     plt.clf()

    t = np.linspace(0, 100, int(100 / 0.01) + 1)
    sol = solve_ivp(lambda t, z: lorenz_dynamics_rescaled(t,z, sigma = 10, beta = 8/3), [t[0], t[-1]], [0, 1, 1.05], t_eval=t)
    ax = plt.figure().add_subplot(projection='3d')
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.2
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.2
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.2
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                        tmp_planes[0], tmp_planes[1],
                        tmp_planes[4], tmp_planes[5])
    view_1 = (25, -135)
    view_2 = (25, -45)
    init_view = view_2
    ax.view_init(*init_view)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    # rotate label
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r"$z$", rotation = 0)

    ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], lw=0.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.show()
    #ax.set_title("Lorenz Attractor")

    #
    #animate(raw_data)