import numpy as np

def grad(x):
    return 2*x - 4

def cost(x):
    return x**2 - 4*x + 5

def myGD1(x0, eta):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(x_new - x[-1]) < 1e-6:
            break
        x.append(x_new)
    return (x, i)

(x1, i1) = myGD1(5, 0.2)
(x2, i2) = myGD1(5, 0.02)

print('solution x1 = %f, cost = %f, after %d iterations' % (x1[-1], cost(x1[-1]), i1))
print('solution x2 = %f, cost = %f, after %d iterations' % (x2[-1], cost(x2[-1]), i2))