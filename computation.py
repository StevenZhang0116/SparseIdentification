import numpy as np


class LorenzSystem:
    def __init__(self, params=None, eta=0):
        """
        the lorentz attractor model

        :param params:  (sigma, beta, rho)
        :param eta:     float, stand deviation of a gaussian noise
        """
        # initial states provided in the paper
        if not params:
            params = (10, 8/3, 28.)
        self.sigma = params[0]
        self.beta = params[1]
        self.rho = params[2]
        self.eta = abs(eta)

    def __call__(self, x):
        dxdt = self.sigma * (x[1] - x[0])
        dydt = x[0] * (self.rho - x[2]) - x[1]
        dzdt = x[0] * x[1] - self.beta * x[2]
        if self.eta > 0:
            dxdt += np.random.normal(0, self.eta)
            dydt += np.random.normal(0, self.eta)
            dzdt += np.random.normal(0, self.eta)
        return np.array([dxdt, dydt, dzdt])


def euler_simulate(dx, x_0, iterations=10**5, dt=1e-3):
    """
    run an euler simulation for any given "function" object
    idk what's going on in here. Probably doesn't need this many np.copy

    :param dx:          a function, or object that supports __call__()
    :param x_0:         initial state given as a n x 1 array
    :param iterations:  int, number of iterations
    :param dt:          float, dt
    :return:            (2dim x iterations) np array as a concatination of
                            a (dim x iterations) x matrix and
                            a (dim x iterations) dx/dt matrix
    """
    x = np.copy(x_0)
    v = np.copy(dx(x_0))
    dim = x.shape[0]
    trajectory = np.zeros((iterations+1, 2 * dim))
    trajectory[0, 0:dim] = x
    trajectory[0, dim:] = v
    for i in range(1, iterations+1):
        x += dt * v
        v = dx(x)
        trajectory[i, 0:dim] = np.copy(x)
        trajectory[i, dim:] = np.copy(v)
    return trajectory[:, :]


def recursive_index(n, p):
    """
    create of list of indices that distribute p power to n elements/dimensions

    :param n: int, dimension of the given data
    :param p: int, polynomial power to be distributed among n dimensions
    :return: a list of index lists [a1, ..., an] such that a1 + ... + an = p
    """
    if n == 1:
        return [[p]]
    else:
        total = []
        for pi in reversed(range(p+1)):
            total += [[pi] + sublist for sublist in recursive_index(n-1, p-pi)]
        return total


# Function objects to output
class PolynomialF:
    def __init__(self, coef, p_distribute):
        self.coef, self.p_distribute = coef, p_distribute

    def __call__(self, x):
        return self.coef * np.prod([x[i] ** p for i, p in enumerate(self.p_distribute)])


class TrigF:
    def __init__(self, coef, n, index):
        self.coef, self.n, self.index = coef, n, index


class SinF(TrigF):
    def __call__(self, x):
        return self.coef * np.sin(self.n * x[self.index])


class CosF(TrigF):
    def __call__(self, x):
        return self.coef * np.cos(self.n * x[self.index])


class MacroSumF:
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, x):
        return sum([f(x) for f in self.fs]) if self.fs else 0


#  =====================================


class SparseIdentification:
    def __init__(self, x, dx, poly_dim=3, trig_dim=2, eta=1e-2, threshold=1e-9):
        self.x = x
        self.dx = dx
        self.dim = x.shape[1]
        self.poly_dim = poly_dim
        self.trig_dim = trig_dim
        self.theta = self.aug_lib(self.poly_dim, self.trig_dim)
        self.f_names = self.generate_f_names()
        self.xi = np.linalg.lstsq(self.theta, dx)[0]
        self.eta = eta
        self.threshold = threshold
        self.representation = []

    def generate_f_names(self):
        """
        No computations going on here. Only string formatting for better display

        :return: list of names of function bases
        """
        f_name_lst = ['1']
        for p in range(1, self.poly_dim+1):
            for p_distribute in recursive_index(self.dim, p):
                f_name = ''
                for i in range(self.dim):
                    if p_distribute[i] == 1:
                        f_name += 'x_' + str(i)
                    elif p_distribute[i] > 1:
                        f_name += 'x_' + str(i) + '^' + str(p_distribute[i])
                f_name_lst.append(f_name)
        for t in range(1, self.trig_dim+1):
            for s in range(self.dim):
                f_name_lst.append('sin(' + str(t) + 'x_'+str(s)+')')
            for s in range(self.dim):
                f_name_lst.append('cos(' + str(t) + 'x_'+str(s)+')')
        return f_name_lst

    def polynomials(self, p):
        """
        create the submatrix in the augmented library where columns have given power p

        :param p: int, power
        :return: [iterations x k_p] matrix
        """
        x_p = [np.prod(self.x ** np.array([p_distribute]), axis=1, keepdims=True)
               for p_distribute in recursive_index(self.dim, p)]
        return np.hstack(x_p)

    def trig(self, t):
        """
        create the submatrix in the augmented library where columns are sin(tx) and cos(tx)

        :param t: int, harmonics
        :return: [iterations x k_t] matrix
        """
        return np.hstack([np.sin(t * self.x), np.cos(t * self.x)])

    def aug_lib(self, p, t):
        """
        augmented library matrix by partitioning the matrix

        :param p: int, the largest degree in polynomial
        :param t: int, largest harmonics for trig
        :return: [iterations x k] matrix
        """
        theta = [self.polynomials(p_) for p_ in range(0, p+1)]
        theta += [self.trig(t_) for t_ in range(1, t+1)]
        return np.hstack(theta)

    def run(self, iterations=20, threshold=None):
        if not threshold:
            threshold = self.threshold
        for k in range(iterations):
            small_ind = np.abs(self.xi) < threshold
            self.xi[small_ind] = 0
            noised = self.dx + self.eta*np.random.normal(0, 1, self.dx.shape)
            for i in range(self.xi.shape[1]):
                big_ind = ~ small_ind[:, i]
                self.xi[big_ind, i] = np.linalg.lstsq(self.theta[:, big_ind], noised[:, i])[0]
        self.xi[np.abs(self.xi) < threshold] = 0
        self.create_representation()
        return self.theta[:, :], self.xi

    def create_representation(self, threshold=0.):
        """
        :param threshold: display only values with magnitude greater than threshold
        :return: None
        """
        xi = np.copy(self.xi)
        xi_int = np.rint(xi)
        xi[np.abs(xi_int - xi) < threshold] = xi_int[np.abs(xi_int - xi) < threshold]
        representation = []
        for i in range(self.dim):
            f_i_repr = ''
            for j in range(self.xi.shape[0]):
                if j == 0 and xi[j, i] != 0:
                    f_i_repr += ' ' + str(xi[j, i]) + ' +'
                elif xi[j, i] > 0:
                    f_i_repr += ' ' + str(xi[j, i]) + ' ' + self.f_names[j] + ' +'
                elif xi[j, i] < 0:
                    f_i_repr = f_i_repr[:-1] + '-' if len(f_i_repr) > 0 else '-'
                    f_i_repr += ' ' + str(-xi[j, i]) + ' ' + self.f_names[j] + ' +'
            f_i_repr = f_i_repr[:-1] if len(f_i_repr) > 0 else 0
            f_i_repr = 'dx_' + str(i) + '/dt =' + f_i_repr
            representation.append(f_i_repr)
        self.representation = representation

    def __str__(self):
        return '\n'.join(self.representation)

    def convert_dx_map(self, threshold=None):
        """
        :param threshold: float, small number
        :return: f:[n]->[n] (a function between lst of size dim to lst of side dim)
        """
        if not threshold:
            threshold = self.threshold
        dx = []
        for i in range(self.dim):
            dxi = []
            j = 0
            for p in range(0, self.poly_dim+1):
                for p_distribute in recursive_index(self.dim, p):
                    if abs(self.xi[j, i]) > threshold:
                        dxi.append(PolynomialF(self.xi[j, i], p_distribute))
                    j += 1
            for n_harm in range(1, self.trig_dim+1):
                for s in range(self.dim):
                    if abs(self.xi[j, i]) > threshold:
                        dxi.append(SinF(self.xi[j, i], n_harm, s))
                    j += 1
                for s in range(self.dim):
                    if abs(self.xi[j, i]) > threshold:
                        dxi.append(CosF(self.xi[j, i], n_harm, s))
                    j += 1
            dx.append(MacroSumF(dxi))
        return lambda x: np.array([dx_i(x) for dx_i in dx])

if __name__ == "__main__":
	# initial conditions
	x_0 = np.array([-8., 7., 27.])
	n = len(x_0)

	# generate data from euler method
	stacked = euler_simulate(LorenzSystem(), x_0)
	np.save('lorentz data.npy', stacked)
	print("original simulation")
	print(stacked)

	# sparse identification
	X = stacked[:, 0:n]
	dX = stacked[:, n:]
	Xi = SparseIdentification(X, dX, 4, 3)
	Xi.run(10)

	# displaying results
	print("computed lorenz system functions")
	print(Xi)
	print("Xi matrix")

	print(Xi.xi)

	# simulate again
	# stacked = euler_simulate(x_0, Xi.convert_dx_map())
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# x_1 = stacked[:, 0]
	# y_1 = stacked[:, 1]
	# z_1 = stacked[:, 2]
	# ax.plot(x_1, y_1, z_1, '.k', markersize=0.5)
	# plt.show()










