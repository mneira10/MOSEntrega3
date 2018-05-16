import network as network
import mnist_loader as mnist_loader
import numpy as np

sigma = 1
mu = 0
sizes = [784, 100, 10]
k = 1.38064852E-23



def mutar(weights, biases):
    global sigma, mu
    dbiases = [sigma*np.random.randn(y, 1) + mu for y in sizes[1:]]
    dweights = [sigma*np.random.randn(y, x) + mu
                for x, y in zip(sizes[:-1], sizes[1:])]
    biases[0] = biases[0] + dbiases[0]
    biases[1] = biases[1] + dbiases[1]
    weights[0] = weights[0] + dweights[0]
    weights[1] = weights[1] + dweights[1]
    return (weights, biases)


def alpha(network, network_prime, T):
    global k
    return np.exp((network.evaluateNormed(test_data)-network_prime.evaluateNormed(test_data))/(T))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


net = network.Network(sizes)
net2 = network.Network(sizes)
best = network.Network(sizes)
net2.cambiarPesos(net)
best.cambiarPesos(net)

arch = open("progreso.dat","a+")
T = 0.10
while(T>0):
    mutar(net.weights, net.biases)
    talpha = alpha(net2,net,T)
    # print("alpha: " + str(talpha))
    if(talpha>np.random.rand()):
        print("entra")
        net2.cambiarPesos(net)
        if(best.evaluateNormed(test_data)<net.evaluateNormed(test_data)):
            best.cambiarPesos(net2)

    arch.write(str(T) + " " + str(best.evaluateNormed(test_data)) + "\n")
    
    T= T-0.0001
    print "Net2: " + str(net2.evaluateNormed(test_data))+ " Net: " + str(net.evaluateNormed(test_data)) + " Best: " + str(best.evaluateNormed(test_data)) + " T: " + str(T) + " alpha: " + str(talpha)

arch.close()

