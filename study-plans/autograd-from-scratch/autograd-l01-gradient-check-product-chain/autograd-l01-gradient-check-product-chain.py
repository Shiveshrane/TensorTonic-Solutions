import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    def L_of(a,b,c,f):
        e=a*b+c
        L=e*f
        return L

    e=a*b+c
    L=L_of(a,b,c,f)
    analytic=(f*b,f*a,f,e)
    numerical=(
        (L_of(a+h, b,c, f)-L)/h,
        (L_of(a, b+h,c, f)-L)/h,
        (L_of(a, b,c+h, f)-L)/h,
        (L_of(a, b,c, f+h)-L)/h,
    )

    
    max_diff=max(abs(x-y) for x, y in zip(analytic, numerical))
    return (float(L),
            [float(x) for x in analytic],
            [float(x) for x in numerical],
            float(max_diff))