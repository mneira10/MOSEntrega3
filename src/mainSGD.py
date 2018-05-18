import network as network
import mnist_loader as mnist_loader
import numpy as np
from random import shuffle

sizes = [784, 100, 10]
iteraciones =30
tam = 10


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def procesarLote(net, lote, eta):
    #gradientes de pesos y biases
    gradB = []
    gradW = []
    #los llenamos con ceros
    for i in range(len(net.biases)):
        gradB.append(np.zeros(net.biases[i].shape))
        gradW.append(np.zeros(net.weights[i].shape))
    #se itera sobre cada imagen del lote
    for im, dig in lote:
        dgradB, dgradW = retropropag(net, im, dig)
        gradB = [nb+dnb for nb, dnb in zip(gradB, dgradB)]
        gradW = [nw+dnw for nw, dnw in zip(gradW, dgradW)]
    #se actualizan los pesos y los biases
    net.weights = [w-(eta/len(lote))*dw
                   for w, dw in zip(net.weights, gradW)]
    net.biases = [b-(eta/len(lote))*db
                  for b, db in zip(net.biases, gradB)]


def retropropag(net, im, dig):
    #se declaran los gradientes de los pesos y los biases
    gradB = []
    gradW = []
    #se llenan de ceros
    for i in range(len(net.biases)):
        gradB.append(np.zeros(net.biases[i].shape))
        gradW.append(np.zeros(net.weights[i].shape))
    #imagen actual 
    act = im
    acts = [im]
    #las respuestas dada una imagen 
    res = []
    #se itera sobre los pesos y los biases
    for w, b in zip(net.weights, net.biases):
        temp = np.dot(w, act) + b
        res.append(temp)
        act = sigmoid(temp)
        acts.append(act)

    d = net.cost_derivative(acts[-1], dig) * sigmoid_prime(res[-1])
    gradB[-1] = d
    gradW[-1] = np.dot(d, acts[-2].transpose())

    for i in range(2, net.num_layers):
        temp = res[-i]
        sp = sigmoid_prime(temp)
        d = np.dot(net.weights[-i+1].transpose(), d)*sp
        gradB[-i] = d
        gradW[-i] = np.dot(d,acts[-i-1].transpose())
    return(gradB,gradW)

#creamos una red
net = network.Network(sizes)
#escribimos resultados a un archivo
arch = open("./dataSGD/data.dat","a+")
print str(0) + " " + str(net.evaluateNormed(test_data))
arch.write(str(0) + " " + str(net.evaluateNormed(test_data)) + "\n")

#iteramos iteraciones veces sobre los datos de entrenamiento
for i in range(1, iteraciones):
    #reorganizamos los datos para un aprendizaje mas eficiente
    shuffle(training_data)
    lotes = []
    #particionamos la información en lotes
    for j in range(0, len(training_data), tam):
        lotes.append(training_data[j:j+tam])
    #procesamos cada lote 
    for lote in lotes:
        procesarLote(net, lote, 3.0)
    #se evalua la eficiencia de la iteracioin
    print str(i) + " " + str(net.evaluateNormed(test_data))
    arch.write(str(i) + " " + str(net.evaluateNormed(test_data)) + "\n")
arch.close()