
from copy import deepcopy

import numpy as np
from scipy.linalg import pinv
from scipy.linalg import qr, qr_delete,  qr_update

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


class ABO:
    '''
    Adaptive Benign Overfitting (using QRD-RLS)
    Model can accommodate both classical and overfitting regimes.
    '''
    def __init__(self, X, y, roll_window, ff, l_reg):

        """
        x - Initial input Dataset - Features (incluing RFFs)
        y - Initial output Dataset - Labels
        roll_window = Rolling window size
        ff = Forgetting factor
        l_reg = lambda(regularization term)
        # beta_t = argmin \sum ff^{t-k} |y_k - X_k beta_t|^2 + ff^T lambda |beta_t|^2
        """

        self.X = X
        self.y = y
        self.dim = len(X)  # number of features
        if ff > 1 or ff <= 0:
            print(f'The forgetting factor {ff} must be in (0,1]')
            return
        self.ff = ff
        self.l_reg = l_reg  # unused for now!

        # Forgetting factor matrix
        ff_sqrt = np.sqrt(ff)
        Bff = np.diag([ff_sqrt ** i for i in range(X.shape[1] - 1, -1, -1)])
        self.Bff = Bff # initialized to roll size
        self.X = self.X @ Bff
        self.y = Bff @ self.y
        self.Q, self.R = qr(self.X.T, check_finite=False)
        self.R_inv = pinv(self.R)
        self.w = self.R_inv @ self.Q.T @ self.y
        # assert np.allclose(self.w, pinv(self.X @ self.X.T) @ self.X @ self.y)
        # only true with ff = 1. Else
        self.roll_window = roll_window
        self.nobs = self.X.shape[1]  # obs in our rolling or expanding window

    @staticmethod
    def extend_row_col(Q: np.array) -> np.array:
        zero_t = np.zeros((Q.shape[0], 1))
        one = np.array([[1]])
        Q = np.block([[Q, zero_t],[zero_t.T, one]])
        return Q

    @staticmethod
    def extend_row(R: np.array) -> np.array:
        row_of_zeros = np.zeros((1, R.shape[1]))
        return np.r_[R, row_of_zeros]

    @staticmethod
    def update_unit(dim):
        return np.vstack((np.zeros((dim, 1)), np.array([[1]])))

    def process_new_data(self,x, y):
        self._update(x,y)

        if self.nobs > self.roll_window:
            # x = self.X[:, 0][:, np.newaxis]
            self._downdate()
        else:
            # expanding window at the start
            pass
        return


    def _update(self, x, y):

        ff_sqrt = np.sqrt(self.ff)
        self.X = np.c_[ff_sqrt * self.X, x]
        self.y = np.r_[ff_sqrt * self.y, y]
        self.R_inv = (1 / ff_sqrt) * self.R_inv
        self.R = ff_sqrt * self.R
        self.Bff = self.extend_row_col(ff_sqrt * self.Bff) # in case increasing window

        update_unit = self.update_unit(self.R.shape[0])
        # update_mat = update_unit @ x.T
        self.Q, self.R = qr_update(self.extend_row_col(self.Q), self.extend_row(self.R),
                                   update_unit, x, check_finite=False)
        self.R_inv = pinv(self.R)  # find fast inv for trapezoidal matrices - Cline 1964
        self.w = self.R_inv @ self.Q.T @ self.y
        self.nobs += 1
        # assert np.allclose(self.w, pinv(self.X @ self.B @  self.X.T) @ self.X @ self.B @ self.y)   put into test!

        #TODO: Awkward as hell! Don't have update call downdate!?!?! Aaargh!


    def _downdate(self):

        """
        downdate first row in X / R / y history
        """
        self.X = self.X[:, 1:]
        self.y = self.y[1:, :]
        self.Bff = self.Bff[1:, 1:] # remove first row and colum on downdate
        self.Q, self.R = qr_delete(self.Q, self.R, k=0, p=1, which='row')
        # assert np.allclose(Q @ R, self.X.T)
        self.R_inv = pinv(self.R)
        self.w = self.R_inv @ self.Q.T @ self.y
        self.nobs -= 1
        # assert np.allclose(self.w, pinv(self.X @ self.X.T) @ self.X @ self.y)

    def pred(self, x):
        """
        x - features (dim x 1)
        """
        if x.shape[1] == 1:
            pred = (x.T @ self.w).item()
        else:
            pred = x.T @ self.w
        return pred

    def get_betas(self):
        betas = self.w
        return np.array(betas.ravel())

    def get_shapes(self):
        return self.X.shape, self.y.shape



# self.all_Q = self.all_Q[1:, 1:]
# self.y = self.y[1:]

# self.R = np.r_[self.R, x.T]
# self.Q, self.R = self.givens_elim(update=True)


# self.Q, _ = self.givens_elim(update=False)
# self.R_inv = self.R_inv @ self.Q
# self.R = self.Q.T @ self.R
# x = self.R[0, :][:, np.newaxis]
# c = np.zeros((self.R_inv.shape[1], 1))
# c[0, 0] = 1
# k = self.R_inv @ c
# h = x.T @ self.R_inv
# #je = x.T @ self.R_inv @ c
#
# # Deletion for new regime
# if not temp:  ## why use the Shearman-Morrison update formula ? Unstable!
#     self.R_inv = (self.R_inv - k @ pinv(k) @ self.R_inv - self.R_inv @ pinv(h) @ h
#                   + (pinv(k) @ self.R_inv @ pinv(h)) * k @ h)
#     # this line causing issues in the QrRLS bagging - k or h are inf
#
# # Deletion for old regime
# else:
#     x = -x
#     h = x.T @ self.R_inv
#     u = (np.eye(self.R_inv.shape[1]) - self.R @ self.R_inv) @ c
#     k = self.R_inv @ c
#     h_mag = h @ h.T
#     u_mag = u.T @ u
#     S = (1 + x.T @ self.R_inv @ c)
#     p_2 = - ((u_mag) / S * self.R_inv @ h.T) - k
#     q_2 = - ((h_mag) / S * u.T - h)
#     sigma_2 = h_mag * u_mag + S ** 2
#     self.R_inv = (self.R_inv +
#                   1 / S * self.R_inv @ h.T @ u.T -
#                   S / sigma_2 * p_2 @ q_2)
#
# self.R_inv = self.R_inv[:, 1:]
# self.R = self.R[1:, :]
# y = self.y
# y = self.all_Q.T @ y
# y = y[1:]
# self.w = self.R_inv @ (self.all_Q.T @ y)[1:]


# update
        # d = x.T @ self.R_inv
        # c = x.T @ (np.eye(self.R.shape[1]) - self.R_inv @ self.R)
        #
        # # Update for new regime, see Cline 1964
        # if not np.allclose(a=c,b=0):
        #     b_k = pinv(c) # i.e., c.T/||c||_2^2
        # # Update for old regime
        # else:
        #     b_k = 1 / (1 + d @ d.T) * self.R_inv @ d.T
        #
        # self.R_inv = np.c_[self.R_inv - b_k @ d, b_k]

        #y =  np.array(self.y).reshape(self.X.shape[1], 1)


    # def givens_elim(self, update=True):
    #
    #     # this section is run if we are updating
    #     if update:
    #         R = self.R
    #         # diag = min(R.shape) - 1
    #         self.all_Q = self.extend_row_col(self.all_Q)  #deepcopy(self.all_Q))
    #         Q, R = qr(R, check_finite=False)
    #         self.all_Q = self.all_Q @ Q
    #         return Q.T, R # wth? Q.T?
    #
    #     # this section is run if we are downdate
    #     else:
    #         q_col = self.all_Q[0, :][:, np.newaxis]
    #         Q, R = qr(q_col, check_finite=False)
    #         self.all_Q = self.all_Q @ Q
    #         return Q, R # wth? here Q?
