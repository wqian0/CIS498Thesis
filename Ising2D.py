import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.linalg import eigh
from copy import copy, deepcopy
import pickle as pk
import random
class Ising2D:
    def __init__(self, L = 64, B = 0.0, beta = 1.0, couplings = None):
        self.B = B
        self.L = L
        self.beta = beta
        self.state = np.array(np.random.randint(0,2,size=(L, L)) * 2 - 1, dtype = float)
        self.couplings = couplings

    def execute(self, steps, use_mc = False, restrictions = None, heatbath = False):
        if use_mc:
            for i in range(steps):
                self.mc_heatbath_update(restrictions = restrictions, heatbath = heatbath)
        else:
            for i in range(steps):
                self.wolff_update()

    def lil_z(self, n_up):
        return np.cosh(2*self.beta * (2 - n_up))

    def neighbor_sum(self, loc):
        s = 0
        for d in self.couplings:
            di, dj = d
            n_loc_i = (loc[0] + di + self.L) % self.L
            n_loc_j = (loc[1] + dj + self.L) % self.L
            n_loc = (n_loc_i, n_loc_j)
            s += self.state[n_loc] * self.couplings[d]
        return s

    def all_neighbor_sums(self):
        nsum = 0
        for d in self.couplings:
            di, dj = d
            shifted = np.roll(self.state, -di, axis = 0)
            shifted = np.roll(shifted, -dj, axis = 1)
            nsum += shifted
        return nsum

    def compute_mf(self):
        nsum = self.all_neighbor_sums()
        p_down_num = np.exp(-self.beta * nsum)
        p_down = p_down_num / (p_down_num + np.exp(self.beta * nsum))
        return 1 - 2* p_down #2d array of meanfield vals

    def group_flip_update(self, restrictions = None):
        if restrictions is None:
            return
        nsum = self.all_neighbor_sums()
        costs = 2 * self.state * (nsum - self.B)
        flip_prob_num = np.exp(-beta * costs)
        flip_prob = flip_prob_num / (flip_prob_num + 1)
        rand_vals = np.random.rand(*self.state.shape)
        self.state[(~restrictions) & (rand_vals < flip_prob)] *= -1

    def higher_level_update(self, allowed, allowed_list):
        rand_idx = np.random.randint(len(allowed_list))
        x_rand = allowed_list[rand_idx]
        z_flip, z_stay = 1, 1
        s = self.state[x_rand]
        for d in self.couplings:
            di, dj = d
            n_loc_i = (x_rand[0] + di + self.L) % self.L
            n_loc_j = (x_rand[1] + dj + self.L) % self.L
            n_loc = (n_loc_i, n_loc_j)
            n_up_curr = (4 + self.neighbor_sum(n_loc)) // 2
            z_flip *= self.lil_z(n_up_curr - s)
            z_stay *= self.lil_z(n_up_curr)
        if np.random.rand() < z_flip / (z_flip + z_stay):
            self.state[x_rand] *= -1

    def mc_heatbath_update(self, restrictions = None, heatbath = False): #randomly select a spin to possibly flip
        x_rand = tuple(np.random.randint(0, self.L, 2))
        if restrictions is not None and restrictions[x_rand] == 1:
            self.mc_heatbath_update(restrictions=restrictions, heatbath = heatbath)
        neighborSum = self.neighbor_sum(x_rand)
        cost = 2 * self.state[x_rand] * (neighborSum - self.B)
        if not heatbath:
            if cost < 0: # net magnetization of neighbors is misaligned with spin at x_rand
                self.state[x_rand] *= -1
            elif np.random.rand() < np.exp(-self.beta * cost): #net magnetization is aligned. Still flips with nonzero probability
                self.state[x_rand] *= -1
        else:
            flip_prob_num = np.exp(-beta*cost)
            flip_prob = flip_prob_num/(flip_prob_num + 1)
            if np.random.rand() < flip_prob:
                self.state[x_rand] *= -1

    def mc_update_allowed(self,allowed_list):
        rand_idx = np.random.randint(len(allowed_list))
        x_rand = allowed_list[rand_idx]
        neighborSum = self.neighbor_sum(x_rand)
        cost = 2 * self.state[x_rand] * (neighborSum - self.B)
        if cost < 0:  # net magnetization of neighbors is misaligned with spin at x_rand
            self.state[x_rand] *= -1
        elif np.random.rand() < np.exp(
                -self.beta * cost):  # net magnetization is aligned. Still flips with nonzero probability
            self.state[x_rand] *= -1

    def wolff_update(self):
        init_loc = tuple(np.random.choice(self.L, 2))
        init_spin = self.state[init_loc]
        cluster = set()
        cluster.add(init_loc)
        stack = [init_loc]
        while stack:
            curr_loc = stack[np.random.randint(len(stack))]
            for d in self.couplings: #neighbors. d is an array with 2 elements
                di, dj = d
                n_loc_i = (curr_loc[0] + di + self.L) % self.L
                n_loc_j = (curr_loc[1] + dj + self.L) % self.L
                n_loc = (n_loc_i, n_loc_j)
                if self.state[n_loc] == init_spin and n_loc not in cluster:
                    p = 1 - np.exp(-2 *self.beta * (self.couplings[d] - self.B * init_spin))
                    if np.random.rand() < p:
                        cluster.add(n_loc)
                        stack.append(n_loc)
            stack.remove(curr_loc)
        for loc in cluster:
            self.state[loc] *= -1

def get_allowed_locs(level, L = 64, force_level_1 = True):
    main_level = get_locs(level = level)
    allowed = ~main_level
    if force_level_1:
        level_1 = get_locs(level=1)
        allowed = allowed & level_1
    allowed_list = []
    for i in range(L):
        for j in range(L):
            if allowed[i][j]:
                allowed_list.append((i,j))
    return allowed, allowed_list

def Phi(state, direction):
    if direction[0] == 0 and direction[1] == 0:
        return np.sum(state)
    di, dj = direction
    shifted = np.roll(state, -di, axis = 0)
    shifted = np.roll(shifted, -dj, axis = 1)
    return np.sum(state * shifted)

def get_locs(L=64, level=0):
    locs = np.zeros((L, L), dtype = bool)
    for i in range(len(locs)):
        for j in range(i, len(locs)):
            if level % 2 == 0:
                locs[i][j] = i % 2 ** (level // 2) == 0 and j % 2 ** (level // 2) == 0
            else:
                locs[i][j] = i % 2 ** ((level - 1) // 2) == 0 and j % 2 ** ((level - 1) // 2) == 0
                locs[i][j] = locs[i][j] and (i + j) % 2 ** ((level - 1) // 2 + 1) == 0
            locs[j][i] = locs[i][j]
    return locs

def coarsened_Phi(state, mf, direction, restrictions, L = 64):
    state = np.array(state, dtype = float)
    state[~restrictions] = mf[~restrictions]
    if direction[0] == 0 and direction[1] == 0:
        return np.sum(state)
    di, dj = direction
    shifted = np.roll(state, -di, axis = 0)
    shifted = np.roll(shifted, -dj, axis = 1)
    combined_prod = state * shifted
    return np.sum(combined_prod)

def compute_FIM(ensemble_size, iterations, couplings, dirs, levels = 5, beta = 1/2.26918531421, B = 0.0, use_mc = False, is_level_1 = False):
    dirs = np.concatenate([[(0,0)], dirs]) #first corresponds to global coupling
    n_params = len(dirs)
    output = np.zeros((n_params, n_params), dtype = np.float64)
    configs = []
    mfs = []
    locs_1 = get_locs(level = 1)
    for i in range(ensemble_size):
        print(i)
        curr_model = Ising2D(L = 64, beta = beta, B = B, couplings = couplings)
        curr_model.execute(iterations,use_mc=use_mc)
        #curr_model.execute(iterations, heatbath=True, restrictions = locs_1)
        configs.append(curr_model.state)
        mfs.append(curr_model.compute_mf())

    phi_table = np.zeros((ensemble_size, n_params))
    for p in range(ensemble_size):
        for u in range(n_params):
            if not is_level_1:
                phi_table[p, u] = Phi(configs[p], dirs[u])
            else:
                phi_table[p,u] = coarsened_Phi(configs[p], mfs[p], dirs[u], locs_1)

    outer_sums = np.zeros((n_params, n_params))
    for u in range(n_params):
        for v in range(u, n_params):
            outer_sums[u,v] = np.sum(np.outer(phi_table[:, u], phi_table[:, v])) #includes terms where q = p. Is subtracted off
            outer_sums[v,u] = outer_sums[u,v]

    for u in range(n_params):
        for v in range(u, n_params):
            output[u,v] += np.dot(phi_table[:,u], phi_table[:,v]) * ensemble_size
            output[u,v] -= outer_sums[u,v]
            output[v,u] = output[u,v]
    output *= 1 / (ensemble_size ** 2 - ensemble_size)

    out_evals = []
    w,v = eigh(output)
    return output, w

def compute_FIM_all(ensemble_size, sub_size, iterations, sub_iters, couplings, dirs, beta = 1/2.26918531421, level_max = 6, B = 0.0, use_mc = False, hybrid_mode = False):
    dirs = np.concatenate([[(0, 0)], dirs])  # first corresponds to global coupling
    n_params = len(dirs)
    output = np.zeros((level_max, n_params, n_params), dtype=np.float64)
    allowed_lists = []
    locs_1 = get_locs(level=1)
    for i in range(level_max - 1):
        force = False
        if hybrid_mode:
            force = True
        _, allowed_list = get_allowed_locs(i + 1, force_level_1=force)
        allowed_lists.append(allowed_list)
    phi_table = np.zeros((level_max, ensemble_size, n_params), dtype = np.float64)
    for i in range(ensemble_size):
        print(i)
        curr_model = Ising2D(L=64, beta=beta, B=B, couplings=couplings)
        curr_model.execute(iterations, use_mc=use_mc)
        curr_state = deepcopy(curr_model.state)
        for l in range(level_max):
            if l == 0:
                for u in range(n_params):
                    phi_table[l][i][u] += Phi(curr_model.state, dirs[u])
            else:
                if hybrid_mode:
                    if l > 1:
                        for j in range(sub_size):
                            for k in range(sub_iters):
                                curr_model.higher_level_update(None, allowed_lists[l - 1])
                            for u in range(n_params):
                                phi_table[l][i][u] += coarsened_Phi(curr_model.state, curr_model.compute_mf(), dirs[u], locs_1)
                    else:
                        for u in range(n_params):
                            phi_table[l][i][u] += coarsened_Phi(curr_model.state, curr_model.compute_mf(), dirs[u], locs_1)
                else:
                    for j in range(sub_size):
                        for k in range(sub_iters):
                            curr_model.mc_update_allowed(allowed_lists[l - 1])
                        for u in range(n_params):
                            phi_table[l][i][u] += Phi(curr_model.state, dirs[u])
                curr_model.state = curr_state
    if hybrid_mode:
        phi_table[2:, :, :] /= sub_size
    else:
        phi_table[1:, :, :] /= sub_size
    for l in range(level_max):
        for u in range(n_params):
            for v in range(u, n_params):
                output[l][u][v] = 1/ensemble_size * np.dot(phi_table[l, :, u], phi_table[l, :, v])
                output[l][u][v] -= np.mean(phi_table[l, :, u]) * np.mean(phi_table[l, :, v])
                output[l][v][u] = output[l][u][v]
    evals = []
    for l in range(level_max):
        w,v = eigh(output[l, :, :])
        evals.append(w)
    return output, evals

def compute_FIM_higher_myway(ensemble_size, sub_size, iterations, sub_iters, couplings, dirs, beta = 1/2.26918531421, level = 2, B = 0.0, use_mc = False):
    dirs = np.concatenate([[(0, 0)], dirs])  # first corresponds to global coupling
    n_params = len(dirs)
    output = np.zeros((n_params, n_params), dtype=np.float64)
    allowed, allowed_list = get_allowed_locs(level, force_level_1=False)
    phi_table = np.zeros((ensemble_size, n_params), dtype = np.float64)
    for i in range(ensemble_size):
        print(i)
        curr_model = Ising2D(L=64, beta=beta, B=B, couplings=couplings)
        curr_model.execute(iterations, use_mc=use_mc)
        for j in range(sub_size):
            for k in range(sub_iters):
                curr_model.mc_update_allowed(allowed_list)
            for u in range(n_params):
                phi_table[i][u] += Phi(curr_model.state, dirs[u])
    phi_table /= sub_size
    for u in range(n_params):
        for v in range(u, n_params):
            output[u][v] = 1/ensemble_size * np.dot(phi_table[:, u], phi_table[:, v])
            output[u][v] -= np.mean(phi_table[:, u]) * np.mean(phi_table[:, v])
            output[v][u] = output[u][v]

    w,v = eigh(output)
    return output, w

def compute_FIM_higher_hybrid(ensemble_size, sub_size, iterations, sub_iters, couplings, dirs, beta = 1/2.26918531421, level = 2, B = 0.0, use_mc = False):
    dirs = np.concatenate([[(0, 0)], dirs])  # first corresponds to global coupling
    n_params = len(dirs)
    output = np.zeros((n_params, n_params), dtype=np.float64)
    allowed, allowed_list = get_allowed_locs(level, force_level_1=True)
    locs_1 = get_locs(level=1)
    phi_table = np.zeros((ensemble_size, n_params), dtype = np.float64)
    for i in range(ensemble_size):
        print(i)
        curr_model = Ising2D(L=64, beta=beta, B=B, couplings=couplings)
        curr_model.execute(iterations, use_mc=use_mc)
        for j in range(sub_size):
            for k in range(sub_iters):
                curr_model.higher_level_update(allowed, allowed_list)
            for u in range(n_params):
                phi_table[i][u] += coarsened_Phi(curr_model.state, curr_model.compute_mf(), dirs[u], locs_1)
    phi_table /= sub_size
    for u in range(n_params):
        for v in range(u, n_params):
            output[u][v] = 1/ensemble_size * np.dot(phi_table[:, u], phi_table[:, v])
            output[u][v] -= np.mean(phi_table[:, u]) * np.mean(phi_table[:, v])
            output[v][u] = output[u][v]

    w,v = eigh(output)
    return output, w

def compute_FIM_higher(ensemble_size, sub_size, iterations, sub_iters, couplings, dirs, beta = 1/2.26918531421, level = 2, B = 0.0, use_mc = False):
    dirs = np.concatenate([[(0,0)], dirs]) #first corresponds to global coupling
    n_params = len(dirs)
    output = np.zeros((n_params, n_params), dtype = np.float64)
    configs = []
    mfs = []
    locs_1 = get_locs(level = 1)
    locs_lev = get_locs(level = level)

    allowed, allowed_list = get_allowed_locs(level)
    for i in range(ensemble_size):
        print(i)
        curr_model = Ising2D(L = 64, beta = beta, B = B, couplings = couplings)
        curr_model.execute(iterations,use_mc=use_mc)
        configs.append([])
        mfs.append([])
        for j in range(sub_size):
            for k in range(sub_iters):
                curr_model.higher_level_update(allowed, allowed_list)
                #curr_model.mc_heatbath_update(restrictions = locs_lev)
            configs[-1].append(curr_model.state)
            mfs[-1].append(curr_model.compute_mf())

    phi_table = np.zeros((ensemble_size, sub_size, n_params))
    for p in range(ensemble_size):
        for r in range(sub_size):
            for u in range(n_params):
                #phi_table[p, r, u] = Phi(configs[p][r], dirs[u])
                phi_table[p, r, u] = coarsened_Phi(configs[p][r], mfs[p][r], dirs[u], locs_1)

    for u in range(n_params):
        for v in range(u, n_params):
            out_sum = np.sum(np.outer(phi_table[:, :, u].flatten(), phi_table[:, :, v].flatten()))
            for q in range(ensemble_size):
                output[u,v] += np.sum(np.outer(phi_table[q, :, u], phi_table[q, :, v])) - np.dot(phi_table[q, :, u], phi_table[q, :, v])
                out_sum -= np.sum(np.outer(phi_table[q, :, u], phi_table[q, :, v])) #get rid of terms where p = q
            for r in range(sub_size):
                out_sum -= np.sum(np.outer(phi_table[:, r, u], phi_table[:, r,v])) #get rid of terms where r = s
            for q in range(ensemble_size):
                for r in range(sub_size):
                    out_sum += phi_table[q,r,u] * phi_table[q,r,v] #add back terms where p = q, r = s
            output[u,v] -= 1/(ensemble_size - 1) * out_sum
            output[u,v] *= 1/(ensemble_size * (sub_size **2 - sub_size))
            output[v,u] = output[u,v]

    w,v = eigh(output)
    return output, w

if __name__ == "__main__":
    couplings = {}
    dirs = [(-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (-2, 1), (-1, 1),(0, 1), (1,1), (1,2), (1,0), (2,0)]
    J = 1
    # for d in dirs:
    #     curr = np.random.rand()*3
    #     couplings[d] = curr
    #     couplings[(-d[0], -d[1])] = couplings[d]

    couplings[(0,1)] = J
    couplings[(1,0)] = J
    couplings[(-1,0)] = couplings[(1,0)]
    couplings[(0,-1)] = couplings[(0,1)]
    beta = 1/2.2691
    #beta = 1

    model = Ising2D(couplings = couplings, beta = beta, B = 0.0)
    for i in range(5):
        model.execute(200)
        plt.imshow(model.state, cmap = 'gray')
        plt.show()

    #print("level 2 hybrid way orig lil_z")
    #fim, evals = compute_FIM(1000, 1000, couplings, dirs, levels = 5, beta = beta,use_mc=False, is_level_1=False)

    #fim, evals = compute_FIM_higher_hybrid(200, 200, 500, 10000, couplings, dirs, beta = beta, level = 2)
    # plt.imshow(fim)
    # print("EVALS", np.sort(evals)[::-1])
    # plt.show()

    # output, evals = compute_FIM_all(150, 150, 600, 10000, couplings, dirs, beta=beta, level_max=6, hybrid_mode=True, B = 0.1)
    # # print(evals)
    # pk.dump([output, evals], open("FIM and evals, hybrid mode, magfield 0_1.pk", "wb"))
    #
    # plt.imshow(fim)
    #
    # print(np.sort(w)[::-1])

    #print(fim)

    # axes = []
    # fig = plt.figure()
    # ax1 = fig.add_subplot(231)
    # ax2 = fig.add_subplot(232)
    # ax3 = fig.add_subplot(233)
    # ax4 = fig.add_subplot(234)
    # ax5 = fig.add_subplot(235)
    # ax6 = fig.add_subplot(236)
    # axes.append(ax1)
    # axes.append(ax2)
    # axes.append(ax3)
    # axes.append(ax4)
    # axes.append(ax5)
    # axes.append(ax6)
    # for i in range(6):
    #     axes[i].title.set_text('Observables, level '+str(i))
    #     if i == 0:
    #         axes[i].imshow(get_locs(level = i), cmap = 'gray')
    #     else:
    #         axes[i].imshow(~get_locs(level=i), cmap='gray')
    # plt.show()






