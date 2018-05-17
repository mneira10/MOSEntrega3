import numpy as np 
import matplotlib.pyplot as plt


dat1 = np.loadtxt("progreso1.dat", delimiter = " ")
dat2 = np.loadtxt("progreso2.dat", delimiter = " ")

plt.plot(dat1[:,1])
plt.plot(dat2[:,1])
plt.xlabel("Iteraciones")
plt.ylabel("Porcentaje de aciertos")
plt.title("Metropolis-Hastings")
# plt.show()
plt.savefig("metropolisHastings.png")
plt.close()