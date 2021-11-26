import numpy as np
import networkx as nx
import Ddist as D

'''
Calculation of D-distance between two networks based on paper:
Schieber, T., Carpi, L., Díaz-Guilera, A. et al. Quantification of network structural dissimilarities. Nat Commun 8, 13928 (2017). https://doi.org/10.1038/ncomms13928
'''

network1 = np.loadtxt('net1.txt', dtype=int)
network2 = np.loadtxt('net2.txt', dtype=int)

#networks should be reindexed: if there are N nodes; nodes Ids should go from 1 to N

#calculate Jensen-Shanon divergence and node distance distributions

JSH1, mu1 = D.calculate_JSH_mu(network1)
d1 = len(mu1) #diametar

JSH2, mu2 = D.calculate_JSH_mu(network2)
d2 = len(mu2) #diametar

JSHM = D.calculate_JSHM(mu1, mu2) #JSH between distance distributions

w1 = 0.5 #global properties - 
w2 = 0.5 #local properties 

Dmeasure=w1*np.sqrt((JSHM)/np.log(2))+w2*np.abs(np.sqrt(JSH1/np.log(d1+1))-np.sqrt(JSH2/np.log(d2+1)))

#Dmeasure takes values from 0 to 1, if it is close to 0 networks are more similar 
print(Dmeasure) 
