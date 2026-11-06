import math


def sigmoid(a):
    if a < -500:
        return 0
    elif a > 500:
        return 1
    return 1/(1 + math.exp(-a))

def hypothesis(w, x):
    a = 0
    for i in range(len(w)):
        a += w[i] * x[i]

    return sigmoid(a)

def cost_single(h, y):
    eps = 1e-15
    h = max(eps, min(1 - eps, h))
    return -y*math.log(h) - (1-y)*math.log(1-h)

def custo(x, y, w):
    m = len(x)
    soma = 0

    for i in range(m):
        h = hypothesis(w, x[i])
        soma += cost_single(h, y[i])

    return soma / m

def update_weights(w, x, y, alpha):
    m = len(x)
    grad = [0] * len(w)

    for i in range(m):
        h = hypothesis(w, x[i])
        error = h - y[i]
        for j in range(len(w)):
            grad[j] += error * x[i][j]

    for j in range(len(w)):
        w[j] = w[j] - alpha * grad[j] / m

    return w

def gradient_descent(w, x, y, alpha, iterations):
    for t in range(iterations):
        w = update_weights(w, x, y, alpha)
        if t % 100 == 0:
            print(t, custo(x, y, w))
    return w


X = [[1, 0.5, 0.5], [1, 1, 1], [1, 3, 2], [1, 2.5, 3]]
y = [0, 0, 1, 1]
w = [0, 0, 0]

print("custo inicial:", custo(X, y, w))
w = gradient_descent(w, X, y, 0.1, 1000)
print("w final:", w)
