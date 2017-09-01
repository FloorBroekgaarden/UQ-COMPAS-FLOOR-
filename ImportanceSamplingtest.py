#IMPORTANCE TEST testing Importance Sampling on COMPAS

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys
import matplotlib
from scipy.stats import multivariate_normal
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D 
from uqPythonSubmit import runGrid
from scipy.integrate import quad

d = 3
#K1 = 0.052075  #normalization constant Kroupa [7,100]
[a_q,b_q] = [0,1]
[a_M1,b_M1] = [7,100] #[K1*7.**(-2.35),K1*(100.)**(-2.35) ]
a_a,b_a = -1,3 # -1 first 

def IMFforNorm(xx):
    return xx**(-2.35)

K1 = 1./(quad(IMFforNorm, a_M1, b_M1)[0])

#N_ini = 100 #initial samples  
#N_ini_ref = 1 #building up Nini
N_ref1 = 80 #Building up IS 
N_ref2 = 10 #building up IS2
f_prior = (0.5)**d
g_M1_ini = (1./(b_M1-a_M1))
alpha = 0.0051#0.5 
         
alpha2 = 0.007  
Error_max = 0.001 #max fraction of error (from real value)  that we want to sample to 


N_needed = []
N_needed_ini = []
REPEAT = 1
Error_residuals = []
N_stops = 0



for l in range(REPEAT):
    print('run %s out of 100'%l)

    N_ini = 0 
    S_ini = []
    TEST = 0
    N_hits_ini = 0
    a_ini_1 = []
    q_ini_1 = []
    M1_ini_1 = []
    FAILS  = 0
    while N_hits_ini < 100: #len(S_ini) <20:
        samples_ini_q = np.random.uniform(a_q, b_q, 1)
        samples_ini_a = 10**np.random.uniform(a_a, b_a,1) #Inverse sampling
        samples_ini_M1 =  np.random.uniform(7, 100,1) # (1./K1)**(1/-2.35) *(np.random.uniform(a_M1,b_M1, 1))**(1/-2.35) # not correct should be IMF: #np.random.uniform(7, 100,1) #
        N_ini = N_ini + 1
       # print(N_ini)
        #print(samples_ini_q)
        #print(samples_ini_a)
        #print(samples_ini_M1)
        
        runGrid('/home/floor/Documents/UQforCOMPAS_2017/CompasUncertaintyQuantification-master/test1',primaryMasses = samples_ini_M1, semiMajorAxes = samples_ini_a, massRatios = samples_ini_q)
        d = np.genfromtxt('test1/allMergers.dat', names=True,skip_header=1)
        if d:
            mergesInHubbleTimeFlag = d['mergesInHubbleTimeFlag']
            if (mergesInHubbleTimeFlag == 1):
                stellarType1 = d['stellarType1']
                if (stellarType1 ==14.0):
                    stellarType2 = d['stellarType2']
                    if (stellarType2 ==14.0):
                        M1ZAMS = d['M1ZAMS']
                        M2ZAMS = d['M2ZAMS']
                        a_ini_1.append(d['separationInitial'])
                        q_ini_1.append(M2ZAMS/float(M1ZAMS))
                        M1_ini_1.append(M1ZAMS)
                        N_hits_ini = N_hits_ini+1
                        print(N_hits_ini)
              #  print('a is %s'%d['separationInitial'])
               # print('q is %s' %(M2ZAMS/float(M1ZAMS)))
               # print('M1 = %s' %M1ZAMS)
       # if not d:
            #print('hello') 
        #    FAILS = FAILS +1
    N_needed_ini.append(N_ini)    
#   print(' N_hits_ini = %s' %N_hits_ini)
    print(M1_ini_1)
#   print(a_ini_1)
#   print(len(q_ini_1))
    print(' total nr of runs = %s' %N_ini)


    #PLOT 
   #  XXA = []
    # YYA = []
    # ZZA = []
    # for i in range(N_ini_1):
     #    XXA.append(S_ini[i][0])
       #  YYA.append(S_ini[i][1])
      #   ZZA.append(S_ini[i][2])
#

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(q_ini_1,a_ini_1,M1_ini_1)
    ax.set_xlabel('q')
    ax.set_ylabel('a')
    ax.set_zlabel('M1')
    plt.title('COMPAS 3D run')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1000)
    ax.set_zlim(7, 100)
    
    np.savetxt('testCOMPAS3.out', np.c_[q_ini_1,a_ini_1,M1_ini_1] )  
    plt.savefig('expl3.pdf', format='pdf', dpi=1200)
    # plt.show()
  #  plt.hist(a_ini_1,50)
  #  plt.show()
    #sigma_1 = 1./(alpha*N_ini) #mostly true for stratified initial sampling
    # cov =  [[sigma_1, 0, 0], [0, sigma_1, 0], [0, 0, sigma_1]] 
   #  samples_ref1 = []
  #   Mixedmean = np.array(S_ini)
  
    frac_importance_sampling =  (1./N_ini)*np.sum(K1*(np.array(M1_ini_1))**(-2.35)/float(g_M1_ini))
    
    print(' the estimated fraction of GWs with IS is %s' %frac_importance_sampling)
print(N_needed_ini)
