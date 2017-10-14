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

#global parameters: 
dimension = 3
N_ini_ref = 100 #building up Nini
N_ref1 = 80 #Building up IS 
N_ref2 = 10 #building up IS2
N_ini_number_ones_to_stop = 30 #sample initially MC until we have this number of succeses
f_prior = (0.5)**dimension 
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
List_errors = []
N_stops = 0
####TEST


def initial_sampling(N_ones_wanted_ini): 

    N_ini = 0 
    S_ini = []
    while len(S_ini) <N_ones_wanted_ini:
        samples_ini = np.random.uniform(low=-1,high=1,size=(N_ini_ref,3))
        N_ini = N_ini + N_ini_ref
        centre1 = d0
        centre2 = d1
        centre3 = d2 
        d1s = np.sqrt(np.sum((samples_ini - centre1)**2,axis=1))
        d2s = np.sqrt(np.sum((samples_ini - centre2)**2,axis=1))
        d3s = np.sqrt(np.sum((samples_ini - centre3)**2,axis=1))
        mask = (d1s < Eps0) | (d2s < Eps1) | (d3s < Eps2)
        if len(S_ini) ==0: #check this
            S_ini = samples_ini[mask]
        else:
            S_ini = np.concatenate((S_ini,samples_ini[mask]),axis =0)        

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
    return N_ini,N_ini_1, S_ini

N_ones_wanted_ini=5
N_ini,N_ini_1,S_ini = initial_sampling(N_ones_wanted_ini)


#Refine1!
sigma_1 = 2*(1./(N_ini**(1./dimension)))**2 #mostly true for stratified initial sampling
cov =  [[sigma_1, 0, 0], [0, sigma_1, 0], [0, 0, sigma_1]] 
samples_ref1 = []
Mixedmean = np.array(S_ini)
    
    

def g(array): #entry is array 
    print('inside g')
    print(N_ini_1,len(array))
    print(array)
    PDFx = np.zeros((N_ini_1,len(array)))
    print(PDFx)
    for i in range(N_ini_1):
        PDFx[i,:] = (multivariate_normal.pdf(list(array),Mixedmean[i],cov))
        print(PDFx)
    MixturePDFx = np.sum(PDFx,axis = 0)*(float(N_ini_1))**-1 
    return MixturePDFx

#OLD:
#def g(array): #entry is array 
#    print('inside g')
#    print(array)
#    PDFx = np.zeros((N_ini_1,len(array)))
#    print(PDFx)
#    for i in range(N_ini_1):
#        PDFx[i,:] = (multivariate_normal.pdf(list(array),Mixedmean[i],cov))
#    print(PDFx)
#    MixturePDFx = np.sum(PDFx,axis = 0)*(float(N_ini_1))**-1 
#    return MixturePDFx

def refine1(N_ini_1,S_ini):  
    '''Refine1: to detect shape '''
    count_outside_parameter_space = 0
    S_ref_all = []
    S_ref_0 = []
    S_0_and_1 =[]
    weights_1 =[]
    N_tot = 0 
    estimator_error = 0.3
    while estimator_error > 0.2:
        for i in range(N_ini_1): 
            sample_ref1_single = np.random.multivariate_normal(S_ini[i], cov, 1) 
    
        #reject if outside [-1,1]**3
            
            if (sample_ref1_single[0][0] < -1) or (sample_ref1_single[0][0] > 1) or (sample_ref1_single[0][1] < -1) or (sample_ref1_single[0][1] > 1) or (sample_ref1_single[0][2] < -1) or (sample_ref1_single[0][2] > 1):
                sample_ref1_single = np.random.multivariate_normal(S_ini[i], cov, 1)
                count_outside_parameter_space += 1 
            #print(i)
            else:
                N_tot += 1 
                #print(N_tot)
                #print(N_tot)
                centre1 = d0
                centre2 = d1
                centre3 = d2 
                #d1s = np.sqrt(np.sum((sample_ref1_single - centre1)**2,axis=1))
               # d2s = np.sqrt(np.sum((sample_ref1_single - centre2)**2,axis=1))
              #  d3s = np.sqrt(np.sum((sample_ref1_single - centre3)**2,axis=1))
              #  mask = (d1s < Eps0) | (d2s < Eps1) | (d3s < Eps2)
                #print('at this moment i is %s'%i)
                #print(i)
                if (np.sqrt(np.sum((sample_ref1_single - centre1)**2,axis=1)) < Eps0) or (np.sqrt(np.sum((sample_ref1_single - centre2)**2,axis=1))< Eps1) or (np.sqrt(np.sum((sample_ref1_single - centre3)**2,axis=1))<Eps2):
                    #print('Floor is here %s' %sample_ref1_single)
                    S_ref_all.append(sample_ref1_single[0])#[mask]
                    S_0_and_1.append(sample_ref1_single[0])
                    print('near gx in once')
                    print(sample_ref1_single[0])
                    print(i)
                    print(Mixedmean[i])
                    gx = multivariate_normal.pdf(sample_ref1_single[0],Mixedmean[i],cov)
                    weights_1.append(gx)
                else:
                    S_ref_0.append(sample_ref1_single[0])
                    S_0_and_1.append(sample_ref1_single[0])

                    #N_ref1_1 +=1
                    #print(S_ref_all)
                #else:
                    #print('Floor is at else %s' %sample_ref1_single)
                 #   S_ref_all = np.concatenate((S_ref_all,sample_ref1_single[mask]),axis =0)
                    #print(S_ref_all)
        MixturePDFx = g(S_ref_all)  #g(Xhat)
        print('Mix PDF = %s,%s'%(len(MixturePDFx),MixturePDFx))
        print('gx s = %s, %s'%(len(weights_1),weights_1))
        Larray = (f_prior)*(MixturePDFx)**-1 #Likelihoods f/g
        MixturePDFx_0 = g(S_ref_0)  #g(Xhat)
        Larray_0 = (f_prior)*((MixturePDFx_0)**-1) #Likelihoods f/g 
        
        
        MixturePDFx_tot = g(S_0_and_1)
        Lall = ((MixturePDFx_tot)**-1)
        
       # print('8 = %s' %((1./N_tot)*sum(Lall)))
        #print('the sum of both weights = %s '%((1./N_tot)*(sum(Larray_0) + sum(Larray))))
        #print('individual weights = %s %s'%(sum(Larray_0),sum(Larray)))
       # print('Ntot = %s, should equal %s + %s'%(N_tot,len(S_ref_all),len(S_ref_0) ))
        Estimator_IS1 = ((N_tot)**-1)*sum(Larray)
        Estimator_IS_weights = sum(Larray) /float(((sum(Larray_0) + sum(Larray))))
        print('EST IS = %s'%Estimator_IS1)
       # print('EST Weights = %s'%Estimator_IS_weights)
        Error_IS1 = abs(fraction_real-Estimator_IS1)/float(fraction_real)
        Error_IS_weights = abs(fraction_real-Estimator_IS_weights)/float(fraction_real)
        print('Error IS ref1 = %s'%Error_IS1)
        #print('Error IS ref1 weights = %s'%Error_IS_weights)
        estimator_error += -0.2
        print('fixed error = %s'%estimator_error)
    
refine1(N_ini_1,S_ini)   
    
    #FIGURE
#XXB = []
#YYB = []
#ZZB = []
#for i in range(len(S_ref_all)):
#    XXB.append(S_ref_all[i][0])
#    YYB.append(S_ref_all[i][1])
#    ZZB.append(S_ref_all[i][2])
#
#fig = plt.figure() 
#ax = fig.add_subplot(111, projection='3d') 
#ax.scatter(XXB,YYB,ZZB) 
#plt.show()   
#    
#    #Refine2
#    N_ref1_ones = len(S_ref_all) #total nr of ones in refined sample
#    S_ref_all2 = []
#    sigma_2 = (1./(N_ini**(1./dimension)))**2 #mostly true for stratified initial sampling
#    cov2 =  [[sigma_2, 0, 0], [0, sigma_2, 0], [0, 0, sigma_2]] 
#    #samples_ref2 = []
#    Mixedmean2 = np.array(S_ref_all)
#    
#    def h(array): #entry is array 
#        PDFx = np.zeros((N_ref1_ones,len(array)))
#        for i in range(N_ref1_ones):
#            PDFx[i,:] = (multivariate_normal.pdf(list(array),Mixedmean2[i],cov2))
#        MixturePDFx2 = np.sum(PDFx,axis = 0)*(float(N_ref1_ones))**-1 
#        return MixturePDFx2
#    
#    k = 0 
#    Error_IS2 = Error_IS1
#    Ntotal = 0
#    while (Ntotal < 9000):  #1/1000 st error 
#        ind_random_Sini2 = np.random.choice(N_ref1_ones, 1,replace=False)[0]
#        S_ref_single2 = np.random.multivariate_normal(S_ref_all[ind_random_Sini2].tolist(), cov2, N_ref2)      
#        centre1 = d0
#        centre2 = d1
#        centre3 = d2 
#        d1s = np.sqrt(np.sum((S_ref_single2 - centre1)**2,axis=1))
#        d2s = np.sqrt(np.sum((S_ref_single2 - centre2)**2,axis=1))
#        d3s = np.sqrt(np.sum((S_ref_single2 - centre3)**2,axis=1))
#        mask = (d1s < Eps0) | (d2s < Eps1) | (d3s < Eps2)
#        if len(S_ref_single2[mask]) >0:
#            if len(S_ref_all2) == 0:
#                S_ref_all2 = S_ref_single2[mask]
#            else: 
#                S_ref_all2 = np.concatenate((S_ref_all2,S_ref_single2[mask]),axis =0) 
#
#        k = k+1
#        Ntotal = N_ini + N_ini_1*N_ref1 + k*N_ref2  
#       #print(Ntotal)
#        if Ntotal == 9000:
#            print('stopped because we reached N')
#            if len(S_ref_all2) >0:
#                MixturePDFx2 = h(S_ref_all2)
#                Larray2 = (f_prior)*(MixturePDFx2)**-1
#                Estimator_IS2 = ((N_ref2*k)**-1)*sum(Larray2) #N_ref2*k is nr of total samples (0s and 1s)
#                Error_IS2 = abs(fraction_real-Estimator_IS2)/float(fraction_real)
#                List_errors.append(Error_IS2)
#                
#        if k == 10**8:
#            print('stopped because we have sampled over all initial ones')
#            N_stops = N_stops+1
#            print('The error when stopped is %s'%Error_IS2)
#            Error_residuals.append(Error_IS2)
#
#        if Error_IS2 <= Error_max: 
#            print('stopped because we reached error')
#     
#    Ntotal = N_ini + N_ini_1*N_ref1 + k*N_ref2   
#    print('total nr of sample points is %s' %Ntotal) 
#    N_needed.append(Ntotal)  
#    
#    XX2 = []
#    YY2 = []
#    ZZ2 = []
#    for i in range(len(S_ref_all2)):
#        XX2.append(S_ref_all2[i][0])
#        YY2.append(S_ref_all2[i][1])
#        ZZ2.append(S_ref_all2[i][2])
#    
#    fig = plt.figure() 
#    ax = fig.add_subplot(111, projection='3d') 
#    ax.scatter(XX2,YY2,ZZ2) 
#
#    plt.title('The output of test function $\phi(x)$: three spheres')
#    plt.show()   
    
#Mean_N_needed = sum(N_needed)*float(len(N_needed)**(-1.))
#print('mean nr of samples needed = %s' %Mean_N_needed)
#print(List_errors)
#print(N_stops)
#print('the errors when stopped are %s' %Error_residuals)


        
