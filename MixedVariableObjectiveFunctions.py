# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
Objective function classes
"""
import numpy as np


class ObjectiveFunction:
    """
    Base class for various objective functions.
    """

    def __init__(self, target_eval=1e-10, max_eval=1e4):
        self.target_eval = target_eval  # Target evaluation value for convergence
        self.max_eval = max_eval  # Maximum number of function evaluations allowed
        self.eval_count = 0  # Number of function evaluations performed so far
        self.best_eval = None  # Best evaluation value found so far

    def __call__(self, X):
        """
        Abstract method for evaluating the objective function for a population X.

        :param X: A population of candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: Evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        raise NotImplementedError("Objective function not implemented!")

    def _update_best_eval(self, evals):
        """
        Update the best evaluation value found so far.

        :param evals: Evaluation values
        :type evals: array_like, shape=(lam), dtype=float
        """
        if self.best_eval is None or evals.min() < self.best_eval:
            self.best_eval = evals.min()


#
class MixedVarsEllipsoid(ObjectiveFunction):
    """
    Unconstrained Ellipsoid function(mixed-integer {x,z}); rounding is enforced on the z-part.
    The function evaluation is executed per a population of size lam stored in a 2D numpy array X.
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, H, c, target_eval=1e-10, max_eval=1e4):
        super(MixedVarsEllipsoid, self).__init__(target_eval, max_eval)
        self.d = d
        self.ind = ind
        self.bid = bid
        self.H = H
        self.c = c
        self.N = self.d // 2

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = np.full(len(X), np.nan)
        for k in range(len(X)):
            y = np.array([X[k, i] for i in range(0, self.N)])  # Real-Valued sub-vector
            z = np.array(
                [np.round(X[k, i]) for i in range(self.N, self.d)])  # Integers sub-vector; rounding is enforced
            xc = np.concatenate((y, z))
            evals[k] = (np.array(xc - c0).dot(self.H).dot(np.array(xc - c0))) / self.c  # The normalized
        self._update_best_eval(evals)
        return evals


#
#
#
"""
Eliipsoids' center generation as a global variable c0
"""


def setC(dim: int, alpha0: int = 7, trans: int = 0):
    global c0
    c0 = trans + np.array([alpha0, -alpha0] * dim)

### EOF ###
