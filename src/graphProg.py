import numpy as np 
import matplotlib.pyplot as plt


dat = np.loadtxt("progreso.dat", delimiter = " ")

plt.plot(dat[:,1])
plt.show()
plt.close()