
from copy import deepcopy

import numpy as np
from scipy.linalg import pinv
from scipy.linalg import qr, qr_delete,  qr_update

class GaussianRFF():

    def __init__(self,features_dim,no_rff,kernel_var=1, seed=True):
        if seed == True:
            np.random.seed(0)
        self.A = np.random.normal(loc=0,scale=kernel_var,size=(features_dim, no_rff)) #normal between 0 and 1
        self.b = np.random.uniform(low=0,high=2*np.pi,size=(1, no_rff)) #uniform between 0 and 2pi
        self.features_dim = features_dim
        self.no_rff = no_rff

    def transform(self, x):
        """
        x - feature vector (T x features_dim)
        A - is features_dim x no_rff
        b - is 1 x no_rff
        z - rff feature vector (T x no_rff)
        z is T x no_rff
        """
        temp = (x  @ self.A  + self.b)
        z = np.sqrt(2 / self.no_rff) * np.cos(temp)
        return z


class ABO(object):
    '''
    Adaptive Benign Overfitting (using QRD-RLS)
    Model can accommodate both classical and overfitting regimes.
    '''
    def __init__(self, X, y, roll_window, ff, l_reg,
                 tests=False):

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
        self.tests = tests
        self.__in_sample_fit = None
        self.__in_sample_resids = None


        # Forgetting factor matrix
        ff_sqrt = np.sqrt(ff)
        Bff = np.diag([ff_sqrt ** i for i in range(X.shape[0] - 1, -1, -1)])
        self.Bff = Bff # initialized to roll size
        self.X = Bff @ self.X
        self.y = Bff @ self.y
        self.Q, self.R = qr(self.X, check_finite=False)
        # X = QR
        # (X'X) = R'R   so (X'X)^+ = R^+ R^+'   X'y = R' Q' y
        # SO (X'X)^+ X'y = R^+ (R^+' R') Q' y
        self.R_inv = pinv(self.R)
        self.w = self.R_inv @ self.Q.T @ self.y

        if self.tests:  # don't calculate otherwise!
            assert np.allclose(self.w, pinv(self.X.T @ self.X) @ self.X.T @ self.y)
        # only true with ff = 1. Else
        self.roll_window = roll_window
        self.nobs = self.X.shape[0]  # obs in our rolling or expanding window
        self.total_num = self.nobs # to start

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

    @property
    def in_sample_resids(self):
        return self.__in_sample_resids

    @property
    def in_sample_fit(self):
        return self.__in_sample_fit


    def _update(self, x, y):

        ff_sqrt = np.sqrt(self.ff)
        self.X = np.r_[ff_sqrt * self.X, x]
        self.y = np.r_[ff_sqrt * self.y, y]
        self.R_inv = (1 / ff_sqrt) * self.R_inv
        self.R = ff_sqrt * self.R
        self.Bff = self.extend_row_col(ff_sqrt * self.Bff) # in case increasing window

        update_unit = self.update_unit(self.R.shape[0])
        # update_mat = update_unit @ x.T
        self.Q, self.R = qr_update(self.extend_row_col(self.Q), self.extend_row(self.R),
                                   update_unit, x.T, check_finite=False)
        self.R_inv = pinv(self.R)  # find fast inv for trapezoidal matrices - Cline 1964
        self.w = self.R_inv @ self.Q.T @ self.y
        if self.tests & (self.total_num < self.nobs + 20):
            assert np.allclose(self.w,
                               pinv(self.X.T @ self.X) @ self.X.T @ self.y)
            # put into test!  now try only for first 20 runs
        self.nobs += 1
        self.total_num +=1

    def _downdate(self):

        """
        downdate first row in X / R / y history
        """
        self.X = self.X[1:, :]
        self.y = self.y[1:, :]
        self.Bff = self.Bff[1:, 1:] # remove first row and colum on downdate
        self.Q, self.R = qr_delete(self.Q, self.R, k=0, p=1, which='row')
        # assert np.allclose(Q @ R, self.X.T)
        self.R_inv = pinv(self.R)
        self.w = self.R_inv @ self.Q.T @ self.y
        self.nobs -= 1
        if self.tests & (self.total_num < self.nobs + 20):
            assert np.allclose(self.w,
                               pinv(self.X.T @ self.X) @ self.X.T @ self.y)

    def in_sample_tests(self):
        '''
        Useful to check if pure interpolant (in non-classical regime) or if
        Returns in_sample_forecasts and resids
        -------
        '''
        self.__in_sample_fit = self.X @ self.w
        self.__in_sample_resids = self.y - self.in_sample_fit

    def pred(self, x):
        """
        x - features (1x dim )
        """
        if x.shape[1] == 1:
            pred = (x @ self.w).item()
        else:
            pred = x @ self.w
        return pred

    def get_betas(self):
        betas = self.w
        return np.array(betas.ravel())

    def get_shapes(self):
        return self.X.shape, self.y.shape


# TODO: Fix dims (X to X.T and vice versa) in following
class ABOBreakpoint(ABO):

    def __init__(self, X, y, roll_window, min_window, ff, l_reg):
        if len(X) > min_window:
            print(f'Warning, len(X) = {len(X)} > min_window = {min_window}. Will truncate')
            X_updates = X[:, min_window:]
            y_updates = y[min_window:, :]
            # truncate and start
            X = X[:, :min_window]
            y = y[:min_window, :]
        super().__init__(X, y, roll_window, ff, l_reg)

        for row_num in X_updates.shape[1]:
            x_row = X[:, row_num]
            y_row = y[row_num,:]
            self.process_new_data(x_row, y_row)

