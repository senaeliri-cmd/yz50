import math
a1 =0.89
a2 = 2.12
w1 = 2.34
w2 = -1.12
b = 1.02


h = 0.0001

def numerical_derivative(x1,x2,x3,x4,x5):
    values = [x1, x2, x3, x4, x5]
    old_res = f(*values)
    r = len(values)

    grads = []

    for i in range(r):
        new_values = values.copy()
        new_values[i] += h
        new_res = f(*new_values)
        loss = (new_res-old_res)/h

        grads.append(loss)
    for i in range(r):
        print(f"{values[i]} grad is :{grads[i]}")


def f(a1, a2, w1, w2, b):
    o = a1 * w1 + a2 * w2 + b
    return tanh(o)

def tanh(x):
    return (math.exp(x*2) - 1)/ (math.exp(2*x) + 1)

numerical_derivative(a1, a2, w1, w2, b)