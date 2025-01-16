import numpy as np

#### Unimodal function
def sphere(x):
    return np.sum(x.flatten()**2)

def schwefel_222(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_12(x):
    return np.sum(np.sum(x[:i+1])**2 for i in range(len(x)))

def quartic_with_noise(x):
    return np.sum([(i+1) * (x[i]**4) for i in range(len(x))]) + np.random.uniform(0,1)


def high_conditioned_elliptic(x):
    # x = x.flatten()
    n = len(x)
    return np.sum([(10**6) ** ((i-1)/(n-1)) * x[i]**2 for i in range(n)])

def rosenbrock(x):
    # x = x.flatten()
    return np.sum([100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1)])

def schwefel_26(x):
    return np.max(np.abs(x.flatten()))

# def hybrid_composition(x, funcs, weights):
#     return np.sum([w * f(x) for f, w in zip(funcs, weights)])

#### Multimodal function
def rastrigin(x):
    A = 10
    return A*len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(x*c))
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a - np.exp(1)

# def griewank(x):
#     sum_1 = 1/4000 * np.exp(x**2) #  np.sum(x**2 / 4000)
#     prod_2 = np.prod(np.cos( x / np.sqrt(np.arange(1, len(x)+1))))
#     return sum_1 - prod_2 + 1

def griewank(x):
    sum_1 = np.sum(x**2) / 4000
    prod_2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return sum_1 - prod_2 + 1

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


#### Hybrid functions
def hybrid_1(x): # aphere + rastrigin + ackley
    n = len(x)
    f1 = sphere(x[:n//3])
    f2 = rastrigin(x[n//3:2*n//3])
    f3 = ackley(x[2*n//3:])
    return f1 + f2 + f3

def hybrid_2(x): # griewank + schwefel + rastrigin
    n = len(x)
    f1 = griewank(x[:n//3])
    f2 = schwefel(x[n//3:2*n//3])
    f3 = rastrigin(x[2*n//3:])
    return f1 + f2 + f3


#### Composition function (weighted sum)
def composition_1(x):
    f1 = sphere(x)
    f2 = rastrigin(x)
    f3 = ackley(x)
    weights = np.array([0.4, 0.3, 0.3])
    return weights[0] * f1 + weights[1] * f2 + weights[2] * f3

def composition_2(x):
    f1 = schwefel(x)
    f2 = griewank(x)
    f3 = ackley(x)
    weights = np.array([0.5, 0.25, 0.25])
    return weights[0] * f1 + weights[1] * f2 + weights[2] * f3