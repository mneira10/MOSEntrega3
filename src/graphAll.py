import numpy as np  
import matplotlib.pyplot as plt 

dataEv = np.loadtxt("metaResults.txt",delimiter=  ' ')
dataStoc1 = np.loadtxt("./dataStochastic/progreso1.dat",delimiter = " " )
dataStoc2 = np.loadtxt("./dataStochastic/progreso2.dat",delimiter = " " )
dataSGD = np.loadtxt("./dataSGD/data1.dat", delimiter = " ")
dataSGD1 = np.loadtxt("./dataSGD/data.dat", delimiter = " ")


plt.plot(dataEv[:,1], label = "max alg. evolutivo",c = "red")
plt.plot(dataStoc1[:,1][:400],label = "Simulated annealing ", c = "blue")
plt.plot(dataStoc2[:,1][:400],c = "blue")
plt.plot(dataSGD[:,1],label = "SGD",c="green")
plt.plot(dataSGD1[:,1],c="green")
plt.xlabel("Iteraciones")
plt.ylabel("Porcentaje de acierto en test_data")
plt.title("Entrenamiento con diversos algoritmos")
plt.legend()
# plt.show()
plt.savefig("todas.png")