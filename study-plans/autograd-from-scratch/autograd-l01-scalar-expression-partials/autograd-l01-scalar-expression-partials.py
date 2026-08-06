import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    def d(a,b,c):
        res=0.0
        res=a*b+c
        return res
    base=d(a,b,c)
    def partial(fp,orig, h):
        return (fp-orig)/h

    da=partial(d(a+h,b,c), base, h)
    db=partial(d(a, b+h,c), base, h)
    dc=partial(d(a,b,c+h), base, h)

    return base, da, db, dc
