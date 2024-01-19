
from copy import deepcopy

import numpy as np
from scipy.linalg import pinv


class GaussianRFF():

    def __init__(self,d,D,kernel_var=1, seed=True):
        if seed == True:
            np.random.seed(0)
        self.A = np.random.normal(loc=0,scale=kernel_var,size=(d,D)) #normal between 0 and 1
        self.b = np.random.uniform(low=0,high=2*np.pi,size=(D,1)) #uniform between 0 and 2pi
        self.D = D

    def transform(self, x):
        """
        x - feature vector (d x 1)
        z - rff feature vector (D x 1)
        """
        temp = (self.A.T @ x + self.b)
        z = np.sqrt(2/self.D) * np.cos(temp)
        return z




class QR_RLS:

    @staticmethod
    def Givens_Rotation(R):
        n = len(R)
        Q = np.identity(n)
        for j in range(R.shape[1]):
            for i in range(j + 1, n):
                x = R[j][j]
                y = R[i][j]
                r = np.sqrt(x ** 2 + y ** 2)
                if r != 0:
                    c = x / r
                    s = -y / r
                    I = np.identity(n)
                    I[i, i] = c
                    I[j, j] = c
                    I[i, j] = s
                    I[j, i] = -s
                    Q = Q @ I.T
                    R = I @ R

        # return R[:n,:]
        return Q, R


    def __init__(self, x, y, max_obs, ff, l):

        """
        x - Initial input Dataset
        y - Initial output Dataset
        max_obs = Rolling window size
        ff = Forgetting factor
        l = lambda(regularization term)
        """

        self.X = x
        self.y = y
        self.dim = len(x)
        self.I = np.eye(self.dim)
        ff = np.sqrt(ff)
        self.ff = ff
        self.l = l
        self.b = 1

        # Forgetting factor matrix
        B = np.diag([ff ** i for i in range(x.shape[1] - 1, -1, -1)])
        self.X = self.X @ B
        self.n_batch = x.shape[1]
        self.Q, self.R = QR_RLS.Givens_Rotation(self.X.T)
        self.R_inv = pinv(self.R)
        self.w = self.R_inv @ self.Q.T @ y
        self.z = self.Q.T @ y

        # A and P were used as R and R inverse
        self.A = self.R
        self.P = self.R_inv
        self.max_obs = max_obs
        self.all_Q = deepcopy(self.Q)
        self.i = 1

    def givens(self, update=True):

        # this section is run if we are updating
        if update:
            A = self.A
            if A.shape[0] > A.shape[1]:
                diag = A.shape[1] - 1

            else:
                diag = A.shape[0] - 1

            G = np.identity(A.shape[0])
            all_Q = deepcopy(self.all_Q)
            all_Q = np.concatenate((all_Q, np.zeros((all_Q.shape[0], 1))), axis=1)
            all_Q = np.concatenate((all_Q, np.zeros((1, all_Q.shape[1]))), axis=0)
            all_Q[-1, -1] = 1
            Q = deepcopy(G)

            for i in range(diag):
                x = A[i, i]
                y = A[-1, i]
                r = np.sqrt(x ** 2 + y ** 2)
                c = x / r
                s = -y / r
                G[i, i] = c
                G[-1, -1] = c
                G[i, -1] = -s
                G[-1, i] = s
                A = G @ A
                Q = Q @ G.T
                G = np.identity(A.shape[0])

            self.all_Q = all_Q @ Q
            return Q.T, A

        # this section is run if we are deleting
        else:

            P = self.P
            G = np.identity(P.shape[1])
            G_all = deepcopy(G)
            diag = P.shape[1] - 1
            A = self.A
            q = self.all_Q[0, :].reshape(self.all_Q.shape[1], 1)

            for i in range(diag, 0, -1):
                x = q[0, 0]
                y = q[i, 0]
                r = np.sqrt(x ** 2 + y ** 2)
                c = x / r
                s = -y / r
                G[i, i] = c
                G[0, 0] = c
                G[0, i] = -s
                G[i, 0] = s
                A = G @ A
                G_all = G_all @ G.T
                q = G @ q
                G = np.identity(self.all_Q.shape[0])

            self.all_Q = self.all_Q @ G_all

            return G_all

    def update(self, x, y):

        self.X = np.c_[self.X, x]
        self.y = np.r_[self.y, y]
        nobs = np.shape(self.X)[1]
        self.P = (1 / self.ff) * self.P
        self.A = self.ff * self.A
        d = x.T @ self.P
        c = x.T @ (np.eye(self.A.shape[1]) - self.P @ self.A)

        # Update for new regime
        if not np.allclose(0, c):
            c_inv = pinv(c)
            self.P = np.c_[self.P - c_inv @ d, c_inv]

        # Update for old regime
        else:
            b_k = 1 / (1 + d @ d.T) * self.P @ d.T
            self.P = np.c_[self.P - b_k @ d, b_k]

        self.A = np.r_[self.A, x.T]
        self.Q, self.A = self.givens()
        y = np.array(self.y).reshape(self.X.shape[1], 1)
        self.w = self.P @ y
        self.P = self.P @ self.Q.T
        self.i += 1

        if nobs > self.max_obs:
            x = self.X[:, 0].reshape(self.dim, 1)
            self.delete(x, self.y[0])

    def delete(self, x, y):

        """
        x - features which will get deleted (dim x 1)
        y - target which will get deleted (scalar)
        """

        temp = np.allclose(np.eye(self.A.shape[1]), self.A.T @ self.P.T)
        self.X = self.X[:, 1:]
        self.Q = self.givens(False)
        self.P = self.P @ self.Q
        self.A = self.Q.T @ self.A
        x = self.A[0, :].reshape(self.dim, 1)
        c = np.zeros((self.P.shape[1], 1))
        c[0, 0] = 1
        k = self.P @ c
        h = x.T @ self.P
        je = x.T @ self.P @ c

        # Deletion for new regime
        if not temp:
            self.P = self.P - k @ pinv(k) @ self.P - self.P @ pinv(h) @ h + (pinv(k) @ self.P @ pinv(h)) * k @ h
            # this line causing issues in the QrRLS bagging - k or h are inf

        # Deletion for old regime
        else:
            x = -x
            h = x.T @ self.P
            u = (np.eye(self.P.shape[1]) - self.A @ self.P) @ c
            k = self.P @ c
            h_mag = h @ h.T
            u_mag = u.T @ u
            S = (1 + x.T @ self.P @ c)
            p_2 = - ((u_mag) / S * self.P @ h.T) - k
            q_2 = - ((h_mag) / S * u.T - h)
            sigma_2 = h_mag * u_mag + S ** 2
            self.P = self.P + 1 / S * self.P @ h.T @ u.T - S / sigma_2 * p_2 @ q_2

        self.P = self.P[:, 1:]
        self.A = self.A[1:, :]
        y = np.array(self.y).reshape(self.X.shape[1] + 1, 1)
        y = self.all_Q.T @ y
        y = y[1:]
        self.w = self.P @ y
        self.all_Q = self.all_Q[1:, 1:]
        self.y = self.y[1:]

    def pred(self, x):
        """
        x - features (dim x 1)
        """

        if x.shape[1] == 1:
            pred = (x.T @ self.w).item()

        else:
            pred = x.T @ self.w
        return pred

