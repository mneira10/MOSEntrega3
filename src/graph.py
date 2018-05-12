import numpy as np
import matplotlib.pyplot as plt 
import time 

while(True):
    data = np.loadtxt("metaResults.txt")
    plt.plot(data[:,0], label = "aveg")
    plt.plot(data[:,1], label = "max")
    plt.plot(data[:,2], label = "min")
    plt.legend()
    plt.xlabel("Generacion")
    plt.ylabel("Fitness normalizado")
    plt.title("Fitness vs Generación")
    # plt.show()
    plt.savefig("results.png")
    plt.close()
    time.sleep(10)