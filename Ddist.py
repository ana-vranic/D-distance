import networkx as nx
import numpy as np


def calculate_JSH_mu(net): #array of connections
    G = nx.Graph()
    G.add_edges_from(net)
    N=len(G)
    P={}
    l=nx.shortest_path_length(G)
    for se in l:
        #print(se)
        i=se[0]
        for j in range(i+1,N+1):
            ln=se[1][j]
            P[i]=P.get(i,{})
            P[j]=P.get(j,{})
            P[i][ln] = P[i].get(ln, 0) +1
            P[j][ln] = P[j].get(ln, 0) +1
                   
    dm=0
    for i in range(1, N+1):
        if (max(float(s) for s in P[i].keys())) > dm:
            dm = max(int(s) for s in P[i].keys())
        for key in P[i]:
            P[i][key] = P[i][key]/(1.0*N)
    
    mu={}
    for l in range(1,dm+1):
        for i in range(1,N+1):
            P[i][l] = P[i].get(l,0)
            mu[l]=mu.get(l,0.0) + P[i][l]   
        mu[l]=mu[l]/(1.0*N)

    JSH=0
    for i in G.nodes():
        for j in range(1,dm+1):
            if P[i][j]>0:
                JSH=JSH+P[i][j]*np.log(P[i][j]/mu[j])
    JSH=JSH/N
    mu_sort = [mu[key] for key in sorted(mu.keys())]
    mu_sort = np.array(mu_sort)
    return [JSH,mu_sort]


def jsd(x,y): #Jensen-shannon divergence - x and y are arrays or lists
    x = np.array(x)
    y = np.array(y)

    d1 = 0.5*x*np.log(2.*x/(x+y))
    d2 = 0.5*y*np.log(2.*y/(x+y))

    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0

    d = np.sum(d1+d2)  

    return d

def calculate_JSHM(a1, a2): #Jensen-shannon divergence - x and y are arrays or lists
    
    dm1 = len(a1)
    dm2 = len(a2)

    if dm1==dm2:
        #print 'dm1=dm2'
        jsm = jsd(a1,a2)
    else: 
        if dm1>dm2:
            #print 'dm1>dm2'
            jsm =  jsd(a1[:dm2],a2) 
            add = a1[dm2:]
            jsm = jsm+np.sum(add*0.5*np.log(2))
        else:
            #print 'dm2>dm1'
            jsm =  jsd(a1, a2[:dm1])
            add = a2[dm1:]
            jsm = jsm+np.sum(add*0.5*np.log(2))
    
    if jsm<0:        
        jsm=0
        
    return jsm  #return float

