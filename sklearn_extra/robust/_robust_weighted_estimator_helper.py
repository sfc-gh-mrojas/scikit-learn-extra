import numba
import numpy as np

from sklearn.utils.extmath import row_norms

import sys
from time import time
from math import exp, log

# Modified from sklearn.cluster._k_means_fast.pyx
np.import_array()

@numba.jit()
def _euclidean_dense_dense(a:float, b:float, n_features:int):
    """Euclidean distance between a dense and b dense"""
    i:int=0
    n:int = n_features // 4
    rem:int = n_features % 4
    result:float = 0

    # We manually unroll the loop for better cache optimization.
    for i in range(n):
        result += ((a[0] - b[0]) * (a[0] - b[0])
                  +(a[1] - b[1]) * (a[1] - b[1])
                  +(a[2] - b[2]) * (a[2] - b[2])
                  +(a[3] - b[3]) * (a[3] - b[3]))
        a += 4; b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result


@numba.jit()
def _kmeans_loss(X,labels):
    """Compute inertia

    squared distancez between each sample and its assigned center.
    """
    dtype = np.double
    n_samples:int = X.shape[0]
    n_features:int = X.shape[1]
    i:int = 0
    j:int = 0
    n_classes:int = len(np.unique(labels))
    centers = np.zeros([n_classes,n_features],dtype = dtype)
    num_in_cluster = np.zeros(n_classes, dtype = int)
    inertias = np.zeros(n_samples, dtype = dtype)
    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]
        num_in_cluster[labels[i]] += 1

    for i in range(n_classes):
        for j in range(n_features):
            centers[i, j] /= num_in_cluster[i]

    for i in range(n_samples):
        j = labels[i]
        inertias[i] = _euclidean_dense_dense(X[i, 0], centers[j, 0], n_features)
    return inertias





# Regression and Classification losses, from scikit-learn.




# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

class LossFunction:
    """Base class for convex loss functions"""
    @numba.jit()
    def loss(self, p:float, y:float):
        """Evaluate the loss function.

        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)

        Returns
        -------
        double
            The loss evaluated at `p` and `y`.
        """
        return 0.

    def py_dloss(self, p:float, y:float):
        """Python version of `dloss` for testing.

        Pytest needs a python function and can't use cdef functions.
        """
        return self.dloss(p, y)

    def py_loss(self, p:float, y:float):
        """Python version of `dloss` for testing.

        Pytest needs a python function and can't use cdef functions.
        """
        return self.loss(p, y)

    @numba.jit()
    def dloss(self, p:float, y:float):
        """Evaluate the derivative of the loss function with respect to
        the prediction `p`.

        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)
        Returns
        -------
        double
            The derivative of the loss function with regards to `p`.
        """
        return 0.


class Regression(LossFunction):
    """Base class for loss functions for regression"""

    def loss(self, p:float, y:float)->float:
        return 0.

    def dloss(self, p:float, y:float)->float:
        return 0.


class Classification(LossFunction):
    """Base class for loss functions for classification"""

    def loss(self, p:float, y:float)->float:
        return 0.

    def dloss(self, p:float, y:float)->float:
        return 0.


class ModifiedHuber(Classification):
    """Modified Huber loss for binary classification with y in {-1, 1}

    This is equivalent to quadratically smoothed SVM with gamma = 2.

    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    @numba.jit()
    def loss(self, p:float, y:float):
        z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return (1.0 - z) * (1.0 - z)
        else:
            return -4.0 * z
    @numba.jit()
    def dloss(self, p:float, y:float)->float:
        z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return 2.0 * (1.0 - z) * -y
        else:
            return -4.0 * y

    def __reduce__(self):
        return ModifiedHuber, ()


class Hinge(Classification):
    """Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    """

    threshold=0.0

    def __init__(self, threshold=1.0):
        self.threshold = Hinge.threshold
    @numba.jit()
    def loss(self, p:float,y:float)->float:
        z = p * y
        if z <= self.threshold:
            return self.threshold - z
        return 0.0
    @numba.jit()
    def dloss(self, p:float, y:float) ->float:
        z = p * y
        if z <= self.threshold:
            return -y
        return 0.0

    def __reduce__(self):
        return Hinge, (self.threshold,)


class SquaredHinge(Classification):
    """Squared Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by
        (quadratically penalized) SVM.
    """

    threshold=0.0

    def __init__(self, threshold=1.0):
        self.threshold = SquaredHinge.threshold
    @numba.jit()
    def loss(self, p:float, y:float) ->float:
        z = self.threshold - p * y
        if z > 0:
            return z * z
        return 0.0
    @numba.jit()
    def dloss(self, p:float, y:float) ->float:
        z = self.threshold - p * y
        if z > 0:
            return -2 * y * z
        return 0.0

    def __reduce__(self):
        return SquaredHinge, (self.threshold,)


class Log(Classification):
    """Logistic regression loss for binary classification with y in {-1, 1}"""
    @numba.jit()
    def loss(self, p:float, y:float)->float:
        z = p * y
        # approximately equal and saves the computation of the log
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z
        return log(1.0 + exp(-z))
    @numba.jit()
    def dloss(self, p:float, y:float) ->float:
        z = p * y
        # approximately equal and saves the computation of the log
        if z > 18.0:
            return exp(-z) * -y
        if z < -18.0:
            return -y
        return -y / (exp(z) + 1.0)

    def __reduce__(self):
        return Log, ()


class SquaredLoss(Regression):
    """Squared loss traditional used in linear regression."""
    @numba.jit()
    def loss(self, p:float, y:float)->float:
        return 0.5 * (p - y) * (p - y)
    @numba.jit()
    def dloss(self, p:float, y:float)->float:
        return p - y

    def __reduce__(self):
        return SquaredLoss, ()


class Huber(Regression):
    """Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    https://en.wikipedia.org/wiki/Huber_Loss_Function
    """

    c=0.0

    def __init__(self, c:float):
        self.c = Huber.c
    @numba.jit()
    def loss(self, p:float, y:float) ->float:
        r = p - y
        abs_r = abs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5 * self.c * self.c)
    @numba.jit()
    def dloss(self, p:float, y:float)->float:
        r = p - y
        abs_r = abs(r)
        if abs_r <= self.c:
            return r
        elif r > 0.0:
            return self.c
        else:
            return -self.c

    def __reduce__(self):
        return Huber, (self.c,)
