from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np

#ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,400., 754., 564., 138., 219., 869., 669.])
#Z = hierarchy.linkage(ytdist, 'single')
plt.figure()
Z=[[0.,9.,3.,6.,11.],[9.,0.,7.,5.,10.],[3.,7.,0.,9.,2.],[6.,5.,9.,0.,8.],[11.,10.,2.,8.,0.]];
Z=np.array(Z)
print(Z)
dn = hierarchy.dendrogram(Z)
plt.show()