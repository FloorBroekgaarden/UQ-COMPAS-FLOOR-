"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:48:28 2017

@author: Floor
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D 

dimension = 3
#N_ini = 500 #initial samples  
N_ini_ref = 100 #building up Nini
N_ref1 = 80 #Building up IS 
N_ref2 = 10 #building up IS2
N_ini_number_ones_to_stop = 30 #sample initially MC until we have this number of succeses
f_prior = (0.5)**dimension 
alpha = 0.0051#0.5 #factor to lower cov such that more ones are created in refinement this is one of the parameters we have to tune
#alpha is probably a factor of Nref1, Nini,         
alpha2 = 0.007  
Error_max = 0.01 #max fraction of error (from real value)  that we want to sample to 
Eps0 = 0.13 #rare event sphere radius # I can pick these random later
Eps1 = 0.13
Eps2 = 0.13
d0 = np.array([0,0,0]) #I can pick these random later
d1 = np.array([-0.2,0.2,0.2])
d2 = np.array([+0.2,-0.17,-0.2])

fraction_real = ((4/3.)*np.pi*Eps0**3)/(2.**3) + ((4/3.)*np.pi*Eps1**3)/(2.**3) + ((4/3.)*np.pi*Eps2**3)/(2.**3)  # + ((4/3.)*np.pi*Epsilon2**3)/(2.**3)
print('the real Volume of ball extreme event with radius Epsilon equals: %s'%fraction_real)
N_needed = []
REPEAT = 1
Error_residuals = []
N_stops = 0
####TEST
for l in range(REPEAT):
    print('run %s out of 100'%l)

    
    #samples_ini = np.random.uniform(low=-1,high=1,size=(N_ini,3))
    N_ini = 0 
    S_ini = []
    while len(S_ini) <N_ini_number_ones_to_stop:
        samples_ini = np.random.uniform(low=-1,high=1,size=(N_ini_ref,3))
        N_ini = N_ini + N_ini_ref
        centre1 = d0
        centre2 = d1
        centre3 = d2 
        d1s = np.sqrt(np.sum((samples_ini - centre1)**2,axis=1))
        d2s = np.sqrt(np.sum((samples_ini - centre2)**2,axis=1))
        d3s = np.sqrt(np.sum((samples_ini - centre3)**2,axis=1))
        mask = (d1s < Eps0) | (d2s < Eps1) | (d3s < Eps2)
#print(np.count_nonzero(mask))
        if len(S_ini) ==0: #check this
            S_ini = samples_ini[mask]
        else:
            S_ini = np.concatenate((S_ini,samples_ini[mask]),axis =0)        
        
        #for i in range(N_ini_ref): #COMPAS does this
         #   if (np.linalg.norm(samples_ini[i]-d0) <Eps0) or (np.linalg.norm(samples_ini[i]-d1)<Eps1) or (np.linalg.norm(samples_ini[i]-d2)<Eps2):
          #          S_ini.append(samples_ini[i])
    
    N_ini_1 = len(S_ini) 
    print('initially we needed %s samples to find %s ones'%(N_ini,N_ini_number_ones_to_stop))
    #PLOT 
    XXA = []
    YYA = []
    ZZA = []
    for i in range(N_ini_1):
        XXA.append(S_ini[i][0])
        YYA.append(S_ini[i][1])
        ZZA.append(S_ini[i][2])
    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(XXA,YYA,ZZA) 
    plt.show()    
    
    
    #Refine1!
    sigma_1 = 2*(1./(N_ini**(1./dimension)))**2 #mostly true for stratified initial sampling
    cov =  [[sigma_1, 0, 0], [0, sigma_1, 0], [0, 0, sigma_1]] 
    samples_ref1 = []
    Mixedmean = np.array(S_ini)
    
    
    ###################################
    #Test to see whether Gaussians are within Sigma space [-1,1]
#    XX = np.random.multivariate_normal([0.6,-0.5], [[sigma_1,0],[0,sigma_1]], N_ref1) 
#    YY = multivariate_normal.pdf(XX,[0.6,-0.5], [[sigma_1,0],[0,sigma_1]])
#    fig = plt.figure() 
#    ax = fig.add_subplot(111, projection='3d') 
#    ax.scatter(XX[:,0],XX[:,1],YY) 
#    plt.show() 
#    
#    Test = lambda x: multivariate_normal.pdf(x,[0.6], sigma_1)
#    print('the total integral within prob space is %s , this should be extremely close to 1,'%scipy.integrate.quad(Test,-1,1)[0])
#    #################################
    def g(array): #entry is array 
        PDFx = np.zeros((N_ini_1,len(array)))
        for i in range(N_ini_1):
            PDFx[i,:] = (multivariate_normal.pdf(list(array),Mixedmean[i],cov))
        MixturePDFx = np.sum(PDFx,axis = 0)*(float(N_ini_1))**-1 
        return MixturePDFx
    
    ind_random_Sini = np.random.choice(N_ini_1, N_ini_1,replace=False) # for taking random a 1 out of the list of initial ones. 
    #create Xhat samples from g: #there should be some while loop 
    
    #Refine1: to detect shape
    S_ref_all = []
    for i in range(N_ini_1): #could be lower and while loop somehow
        S_ref_single = np.random.multivariate_normal(S_ini[ind_random_Sini[i]], cov, N_ref1) 
 #       samples_add = np.random.uniform(low=-1,high=1,size=(N_add,3)) #1000 new random samples
#
        centre1 = d0
        centre2 = d1
        centre3 = d2 
        d1s = np.sqrt(np.sum((S_ref_single - centre1)**2,axis=1))
        d2s = np.sqrt(np.sum((S_ref_single - centre2)**2,axis=1))
        d3s = np.sqrt(np.sum((S_ref_single - centre3)**2,axis=1))
        mask = (d1s < Eps0) | (d2s < Eps1) | (d3s < Eps2)
#print(np.count_nonzero(mask))
        if i == 0:
            S_ref_all = S_ref_single[mask]
        else:
            S_ref_all = np.concatenate((S_ref_all,S_ref_single[mask]),axis =0)

      #  for j in range(N_ref1): #COMPAS does this
       #     if (np.linalg.norm(S_ref_single[j]-d0) <Eps0) or (np.linalg.norm(S_ref_single[j]-d1)<Eps1) or (np.linalg.norm(S_ref_single[j]-d2)<Eps2):
        #        S_ref_all.append(S_ref_single[j])#is normalized
                
       
    #S_all_ini =  S_ref_all + S_ini
    
    MixturePDFx = g(S_ref_all)  #g(Xhat)
    Larray = (f_prior)*(MixturePDFx)**-1 #Likelihoods f/g
    
    Estimator_IS1 = ((N_ref1*N_ini_1)**-1)*sum(Larray)
    #print('the IS estimator is: %s' %Estimator_IS1)
    Error_IS1 = abs(fraction_real-Estimator_IS1)/float(fraction_real)
    #print('the error with IS is: %s procent'%Error_IS1)
    #FIGURE
    XXB = []
    YYB = []
    ZZB = []
    for i in range(len(S_ref_all)):
        XXB.append(S_ref_all[i][0])
        YYB.append(S_ref_all[i][1])
        ZZB.append(S_ref_all[i][2])
    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(XXB,YYB,ZZB) 
    plt.show()   
#    
    #Refine2
    N_ref1_ones = len(S_ref_all) #total nr of ones in refined sample
    S_ref_all2 = []
    sigma_2 = (1./(N_ini**(1./dimension)))**2 #mostly true for stratified initial sampling
    cov2 =  [[sigma_2, 0, 0], [0, sigma_2, 0], [0, 0, sigma_2]] 
    #samples_ref2 = []
    Mixedmean2 = np.array(S_ref_all)
    
    def h(array): #entry is array 
        PDFx = np.zeros((N_ref1_ones,len(array)))
        for i in range(N_ref1_ones):
            PDFx[i,:] = (multivariate_normal.pdf(list(array),Mixedmean2[i],cov2))
        MixturePDFx2 = np.sum(PDFx,axis = 0)*(float(N_ref1_ones))**-1 
        return MixturePDFx2
    
    
    #N_ref_samples = 10
    k = 0 
    Error_IS2 = Error_IS1
    while (Error_IS2 > Error_max) and (k < 10**6):  #1/1000 st error 
        ind_random_Sini2 = np.random.choice(N_ref1_ones, 1,replace=False)[0]
        S_ref_single2 = np.random.multivariate_normal(S_ref_all[ind_random_Sini2].tolist(), cov2, N_ref2)      
        centre1 = d0
        centre2 = d1
        centre3 = d2 
        d1s = np.sqrt(np.sum((S_ref_single2 - centre1)**2,axis=1))
        d2s = np.sqrt(np.sum((S_ref_single2 - centre2)**2,axis=1))
        d3s = np.sqrt(np.sum((S_ref_single2 - centre3)**2,axis=1))
        mask = (d1s < Eps0) | (d2s < Eps1) | (d3s < Eps2)
#print(np.count_nonzero(mask))
        #print(S_ref_single2[mask])
        if len(S_ref_single2[mask]) >0:
            if len(S_ref_all2) == 0:
                S_ref_all2 = S_ref_single2[mask]
            else: 
                S_ref_all2 = np.concatenate((S_ref_all2,S_ref_single2[mask]),axis =0) 
        #print(S_ref_all2)
     #else:
      #      S_ref_all2 = np.concatenate((S_ref_all2,S_ref_single2[mask]),axis =0) 
 #      for j in range(N_ref2):
  #          if (np.linalg.norm(S_ref_single2[j]-d0) <Eps0) or (np.linalg.norm(S_ref_single2[j]-d1)<Eps1) or (np.linalg.norm(S_ref_single2[j]-d2)<Eps2):
   #             S_ref_all2.append(S_ref_single2[j]) #is normalized
        k = k+1
        #print(k)
        if k == 10**8:
            print('stopped because we have sampled over all initial ones')
            N_stops = N_stops+1
            print('The error when stopped is %s'%Error_IS2)
            Error_residuals.append(Error_IS2)
        if len(S_ref_all2) >0:
            MixturePDFx2 = h(S_ref_all2)
            Larray2 = (f_prior)*(MixturePDFx2)**-1
            Estimator_IS2 = ((N_ref2*k)**-1)*sum(Larray2) #N_ref2*k is nr of total samples (0s and 1s)
            Error_IS2 = abs(fraction_real-Estimator_IS2)/float(fraction_real)
       # print(Error_IS2)
        if Error_IS2 <= Error_max: 
            print('stopped because we reached error')
     
    Ntotal = N_ini + N_ini_1*N_ref1 + k*N_ref2   
    print('total nr of sample points is %s' %Ntotal) 
    N_needed.append(Ntotal)  
    
    XX2 = []
    YY2 = []
    ZZ2 = []
    for i in range(len(S_ref_all2)):
        XX2.append(S_ref_all2[i][0])
        YY2.append(S_ref_all2[i][1])
        ZZ2.append(S_ref_all2[i][2])
    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(XX2,YY2,ZZ2) 
    plt.show()   
    
Mean_N_needed = sum(N_needed)*float(len(N_needed)**(-1.))
print('mean nr of samples needed = %s' %Mean_N_needed)
print(N_stops)
print('the errors when stopped are %s' %Error_residuals)


        
