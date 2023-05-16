import numpy as np
from scipy.optimize import minimize

from hydraPlus import hydra


class HydraPlus:
    eps = np.finfo(np.double).eps

    def __init__(
        self, dists, dim, curvature=-1.0, alpha=1.1, equi_adj=0.5, max_iter=1000
    ):
        self.dists = dists
        self.dim = dim
        self.curvature = curvature
        self.n_taxa = len(dists)
        self.alpha = alpha
        self.equi_adj = equi_adj
        self.max_iter = max_iter

    def curve_embed(self):
        """Search for the optimal curvature, then embed.

        Find the curvature giving the least embedding stress from Hydra+. Using
        gradient-free Nelder-Mead method since the stress is too non-convex.

        Whilst, it's possible to run the location and curvature jointly. Better
        results seemling from running them seperately.
        """
        print(f"Initial curvature given {self.curvature}.")
        print("Optimising to given distances:", flush=True)
        optimizer = minimize(
            self.get_stress_curvature,
            self.curvature,
            # jac=self.dstress_dcurvature,
            method="Nelder-Mead",
            bounds=[(None, 0.0)],
            options={"disp": False, "maxiter": self.max_iter},
        )
        print("Local optima found.")
        print(f"Setting curvature to: {optimizer.x[0]}", flush=True)
        return self.embed()

    def get_stress_curvature(self, curvature):
        self.curvature = curvature
        emm = self.embed()
        self.locs = emm["X"]
        return emm["stress_hydraPlus"]

    def embed(self):
        """Embed the distance matrix into the Hyperboloic sheet using Hydra+.

        Use hydra to initialise points, then run BFGS minimiser to minimise the
        embedding stresss.
        """
        emm = hydra.hydra(
            self.dists,
            self.dim,
            curvature=self.curvature,
            alpha=self.alpha,
            equi_adj=self.equi_adj,
            stress=True,
        )
        loc_poin = np.tile(emm["r"], (self.dim, 1)).T * emm["directional"]
        loc_hyp_sheet = self.poincare_to_sheet(loc_poin)
        loc_hyp_exact = self.sheet_to_exact(loc_hyp_sheet).flatten()

        optimizer = minimize(
            self.get_stress_x,
            loc_hyp_exact,
            method="BFGS",
            jac=self.dstress_dx,
            options={"disp": False},
        )
        final_exact = optimizer.x.reshape((self.n_taxa, self.dim))

        output = {}
        output["X"] = final_exact
        output["stress_hydra"] = emm["stress"]
        output["stress_hydraPlus"] = optimizer.fun
        output["curvature"] = self.curvature
        output["dim"] = self.dim
        return output

    def dstress_dx(self, x):
        """Calculates the gradient for stress-minimzation.
        Gradient of stress wrt embedding locations x.

        Args:
            x (ndarray): the vectorization of the coordinate matrix X, which
            has dimensions nrows x ncols. The rows of X are the embedded points
            and the columns the reduced hyperbolic coordinates

        Returns:
            ndarray: The flattened Jacobian.
        """
        x = x.reshape((self.n_taxa, self.dim))
        x = self.exact_to_sheet(x)
        X = np.matmul(x, x.T)
        u_tilde = np.sqrt(X.diagonal() + 1)
        H = X - np.outer(u_tilde, u_tilde)
        H = np.minimum(H, -(1 + self.eps))
        D = 1 / np.sqrt(-self.curvature) * np.arccosh(np.maximum(-H, 1))
        np.fill_diagonal(D, 0)
        A = (D - self.dists) * (1 / np.sqrt(-self.curvature * (H**2 - 1)))
        np.fill_diagonal(A, 0)
        B = np.outer((1 / u_tilde), u_tilde)
        AB_sum = np.tile(np.sum(A * B, axis=1), (self.dim + 1, 1)).T
        G = 2 * (AB_sum * x - A @ x)
        G = self.sheet_to_exact(G)
        return G.flatten()

    def dstress_dcurvature(self, curvature):
        """Calculates the gradient for stress-minimzation.
        Gradient of stress wrt curvature.

        Args:
            x (nparray): flattened hyperbolic coordinates
                         (omitting the first coordinate).
        """
        x = self.locs.reshape((self.n_taxa, self.dim))
        x = self.exact_to_sheet(x)
        X = np.matmul(x, x.T)
        u_tilde = np.sqrt(X.diagonal() + 1)
        H = X - np.outer(u_tilde, u_tilde)
        H = np.minimum(H, -(1 + self.eps))
        D = 1 / np.sqrt(-curvature) * np.arccosh(np.maximum(-H, 1))
        np.fill_diagonal(D, 0)
        g = -0.5 * sum(
            sum(0.5 * (-curvature) ** (-1.5) * (D - self.dists) * np.arccosh(np.maximum(-H, 1)))
        )
        return g

    def get_stress_x(self, x):
        x = x.reshape((self.n_taxa, self.dim))
        x = self.exact_to_sheet(x)
        X = np.matmul(x, x.T)
        u_tilde = np.sqrt(X.diagonal() + 1)
        H = X - np.outer(u_tilde, u_tilde)
        D = 1 / np.sqrt(-self.curvature) * np.arccosh(np.maximum(-H, 1))
        np.fill_diagonal(D, 0)
        y = 0.5 * np.sum((D - self.dists) ** 2)
        return y

    def exact_to_sheet(self, loc):

        z = np.expand_dims(
            np.sqrt(np.sum(np.power(loc, 2), 1) / (self.curvature**2 + self.eps) + 1), 1
        )
        return np.concatenate((z, loc), axis=1)

    @staticmethod
    def sheet_to_exact(loc):
        return loc[:, 1:]

    def poincare_to_sheet(self, loc_poin):
        a = np.power(loc_poin, 2).sum(axis=-1)
        out0 = (1 + a) / (1 - a)
        out1 = 2 * loc_poin / (1 - np.expand_dims(a, axis=1) + self.eps)
        return np.concatenate((np.expand_dims(out0, axis=1), out1), axis=1)
