import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    def horner(x):
        res=0.0
        for c in reversed(coefficients):
            res=res*x+c
        return res

    fx=horner(x)
    fxh=horner(x+h)
    return fx, fxh, (fxh-fx)/h
