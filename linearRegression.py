import random

def randomW():
    return random.uniform(1.0, 10.0)

def gerar_dados(m):
    y = []
    x = []

    for i in range(m):
        x1 = randomW()
        y1 = 1 + 2*x1 + random.gauss(0, 1)

        x.append(x1)
        y.append(y1)

    return x, y

def custo(w0, w1, x, y):
    m = len(x)
    soma = 0

    for i in range(m):
        prevision = w0 + w1*x[i]
        error = (prevision - y[i])**2
        soma = soma + error

    return 1/(2*m) * soma

def update_weights(w0, w1, x, y, alpha):
    m = len(x)
    soma0 = 0
    soma1 = 0

    for i in range(m):
        prevision = w0 + w1*x[i]
        error = (prevision - y[i])
        soma0 = soma0 + error
        soma1 = soma1 + error * x[i]

    new0 = w0 - (1*(alpha/m) * soma0)
    new1 = w1 - (1*(alpha/m) * soma1)

    return new0, new1

def gradiente_descent(x, y, alpha, iterations):
    w0 = randomW()
    w1 = randomW()

    for i in range(iterations):
        w0, w1 = update_weights(w0, w1, x, y, alpha)
        if i % 100 == 0:
            print(i, custo(w0, w1, x, y))


    return w0, w1


x, y = gerar_dados(50)
print(gradiente_descent(x, y, 0.01, 1000))
