'''Adaptive importance sampling algorithm for paper '''
#Import functions
from __future__ import division 
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

#parameter ranges:
[a_q,b_q] = [0,1] 
[a_M1,b_M1] = [7,100] #in Msun 
[a_a,b_a] = -1,3 #in AU and Log10
dimension = 3


#pdfs initial parameters 
def IMFforNorm(xx):
    return xx**(-2.35)
K1 = 1./(quad(IMFforNorm, a_M1, b_M1)[0])
def prior_M1(x):
    alpha_local = 2.35
    prob_M1 = K1*x**(-alpha_local)
    return prob_M1

def prior_a(x):
    prob_a = (1./(x*np.log(10.)))*(1./(b_a-a_a))
    return prob_a

def prior_q(x):
    prob_q = 1./(b_q-a_q)
    return prob_q


def testpriorIMF1(x):
    return multivariate_normal.pdf(x,mean = 40,cov = [800])

def testpriorIMF2(x):
    return multivariate_normal.pdf(x,mean = 60,cov = [800])

#print quad(testpriorIMF,  7,100)[0], 'integral'
A1 = 1./(quad(testpriorIMF1,  7,100)[0])
A2 = 1./(quad(testpriorIMF2,  7,100)[0])
K2 = [A1,A2]
#print quad(testpriorIMF,  -10000,7)[0], 'integral'
#print quad(testpriorIMF,  100,1000)[0], 'integral'

#Sampling from pdf
def sampling_from_IMF(Nsamples): 
    '''returns  N samples distributed to IMF with alpha_local using inverse sampling'''
    alpha_local = 2.35
    samples = ((np.random.uniform(0,1, Nsamples))*(100**(1.-alpha_local) - 7**(1.-alpha_local)) +7**(1-alpha_local))**(1./(1.-alpha_local))
    return samples

def sampling_from_a(Nsamples):
    '''returns  N samples distributed log uniform via inverse sampling'''
    return 10**np.random.uniform(a_a, b_a,Nsamples)

def sampling_from_q(Nsamples):
    '''returns  N samples distributed uniform'''
    return np.random.uniform(a_q, b_q, Nsamples)

def initial_sampling(N_ones_wanted_ini):
    '''takes samples and evaluates them until N_ones_wanted_ini successful (i.e. hits) samples are found'''
    print('initial sampling starts')
    N_hits_ini = 0
    N_ini_all = 0
    #make lists for parameters from model
    a_ini_1,q_ini_1,M1_ini_1 = [],[],[]
    M1_BH,M2_BH = [],[]
    optimisticCEFlag,RLOFSecondaryAfterCEE = [],[] #check this with Jim Barrett
    gaus_mean_for_ref1 = []
    a_ini_all,q_ini_all,M1_ini_all = [],[],[]
    weights_ini_1 =[]
    A =0
    count_outside_parameter_space_ini = 0
    mean = [40,60]
    mean_a = [2,5]
    #run new initial samples for COMPAS until certain number of successful sample runs are found. 
    while N_hits_ini < N_ones_wanted_ini: 
        for i in range(2):

            samples_ini_M1 =  multivariate_normal.rvs(mean = mean[i], cov = [800],size = 1) #np.random.uniform(7, 100,1) #  sampling_from_IMF(1) #change to IS: 
            samples_ini_a =  sampling_from_a(1) 
            samples_ini_q = sampling_from_q(1)
            
            M1_ini_all.append(samples_ini_M1)
            a_ini_all.append(samples_ini_a)
            q_ini_all.append(samples_ini_q)
            
            if (samples_ini_M1 >100) or (samples_ini_M1 <7):
                count_outside_parameter_space_ini +=1
                N_ini_all = N_ini_all + 1
                print 'outside parameterspace', count_outside_parameter_space_ini, samples_ini_M1
            elif A ==0:
                N_ini_all = N_ini_all + 1
                N_hits_ini = N_hits_ini+1 
               # print M1_ini_1
                M1_ini_1.append(samples_ini_M1)
                #print 'here', samples_ini_M1
                gaus_mean_for_ref1.append([samples_ini_M1,samples_ini_a[0],samples_ini_q[0]])
                #print prior_M1(samples_ini_M1[-1])
                #print ' Hallo Jeroen ik ben hier' #K2[i]*
                w_temp = prior_M1(samples_ini_M1)/(multivariate_normal.pdf(samples_ini_M1,mean = mean[i],cov = [800]))  #(1./(b_M1-a_M1))
               # print w_temp
              #  print type(w_temp)
                weights_ini_1.append(w_temp)

            #print weights_ini_1
          #  print(N_hits_ini)
        print 'IS estimator 2 = ', (1./(N_ini_all))*sum(weights_ini_1)
            #print 'IS estimator = ', (1./(N_ini_all))*sum(weights_ini_1)
    print count_outside_parameter_space_ini, 'outside' 
    return N_ini_all,M1_ini_1,weights_ini_1, gaus_mean_for_ref1




def MonteCarlo_estimator(N_tot,N_ones):
    return (1./N_tot) * N_ones 

def ImportanceSampling_estimator_ini(N_tot_samples,weights):
    print weights 
    print sum(weights)
    ISestimator_ini =(1./N_tot_samples)*sum(weights) 
    return ISestimator_ini 




def ImportanceSampling_estimator_ref1(N_ref1_all,weights_temp):
    #print 'len 1 + len 0 = ', len(weights_0), len(weights_1), 'should equal', N_ref1_all
    ISestimator =  (1./(N_ref1_all))*sum(weights_temp) #* (1./(float(sum(weights_0))+sum(weights_1))) #(1./N_ref1_all)*
    return ISestimator



def refined_1_sampling(N_ini_ones_choice,gaus_mean_for_ref1,N_ini_all):
    ''' '''
    print('refinement 1 starts')
    #define the covariance matrix for the gaussians by sigma = average distance between two hits. 
    sigma_squared_M1 = 1*(abs(a_M1-b_M1)/(N_ini_all**(1./dimension)))**2 #factor two to make it a little larger
    sigma_squared_a = 1*(abs(a_a-b_a)/(N_ini_all**(1./dimension)))**2 #whatch out! used to be abs(a_a-b_a)
    sigma_squared_q = 1*(abs(a_q-b_q)/(N_ini_all**(1./dimension)))**2
    cov =  [[sigma_squared_M1, 0, 0], [0, sigma_squared_a, 0], [0, 0, sigma_squared_q]]
    print 'the cov matrix = ', cov
    #make lists for /set parameters
    N_ref1_all = 0 
    a_ref1_1,q_ref1_1,M1_ref1_1 = [],[],[]
    M1_ref1_BH,M2_ref1_BH = [],[]
    optimisticCEFlag_ref1,RLOFSecondaryAfterCEE_ref1 = [],[]
    a_ref1_all,q_ref1_all,M1_ref1_all = [],[],[]
    samples_all_ref1 = []
    samples_1_ref1 = []
    weights_1_for_IS_estimator_ref1 = []
    weights_0_for_IS_estimator_ref1 = []
    # for ref1
    estimator_error =2000.5 
    count_outside_parameter_space = 0 
    
    while estimator_error > 0.2:
        ''' '''
        for i in range(N_ini_ones_choice): #Jim Barrett; can we do this faster (for loop)?
            #print gaus_mean_for_ref1[i]
            #print cov 
            #print np.random.multivariate_normal(gaus_mean_for_ref1[i], cov, 1)
            A =0
          #  print 'gaus mean = ',gaus_mean_for_ref1[i], len(gaus_mean_for_ref1[i])
            sample_ref1_single = multivariate_normal.rvs( mean = gaus_mean_for_ref1[i], cov = cov,size = 1) #np.random.multivariate_normal(gaus_mean_for_ref1[i], cov, 1)[0]  # take one sample from Gauss around one mean do this for all i
            #rejection sampling: 
            #while
            samples_all_ref1.append(sample_ref1_single)
            M1_ref1_all.append(sample_ref1_single[0]) 
            a_ref1_all.append(sample_ref1_single[1])
            q_ref1_all.append(sample_ref1_single[2]) 
            
            if (sample_ref1_single[0] < a_M1) or (sample_ref1_single[0] > b_M1) or (sample_ref1_single[1] < 10.**a_a) or (sample_ref1_single[1] > 10.**b_a) or (sample_ref1_single[2] < a_q) or (sample_ref1_single[2] > b_q):
                #sample_ref1_single = multivariate_normal.rvs( mean = gaus_mean_for_ref1[i], cov = cov,size = 1) #np.random.multivariate_normal(gaus_mean_for_ref1[i], cov, 1)[0]
                count_outside_parameter_space += 1
                N_ref1_all +=1

            #print sample_ref1_single

            else:
            #print 'sample ref single', sample_ref1_single
            #run compas with not rejected sample point
                if A ==0:
                    #print 'hello2'

                    M1_ref1_1.append(sample_ref1_single[0])
                    a_ref1_1.append(sample_ref1_single[1])
                    q_ref1_1.append(sample_ref1_single[2])

                    samples_1_ref1.append([sample_ref1_single[0],sample_ref1_single[1],sample_ref1_single[2]]) #this are the samples combined as (M1,a,q)


                    prior_sample_1 = prior_M1(float(sample_ref1_single[0]))*prior_a(float(sample_ref1_single[1]))*prior_q(sample_ref1_single[2]) 
                    instrumental_sample_1 = (multivariate_normal.pdf(sample_ref1_single,gaus_mean_for_ref1[i],cov))
                    #instrumental_sample_1 = calculates_mixturePDF(N_ini_ones_choice,gaus_mean_for_ref1,samples_1_ref1[-1],cov)
                    #print('IS prob')
                    weights_1_for_IS_estimator_ref1.append(prior_sample_1/float(instrumental_sample_1))
                    #print(weights_1_for_IS_estimator_ref1[-1])
                    N_ref1_all +=1  
                        #print 'at GW', N_ref1_all
                    #else:
                     #   prior_sample_0 = 1 #prior_M1(float(sample_ref1_single[0]))*prior_a(float(sample_ref1_single[1]))*prior_q(float(sample_ref1_single[2])) 
                        #instrumental_sample_0 = (multivariate_normal.pdf(sample_ref1_single,gaus_mean_for_ref1[i],cov))
                      #  instrumental_sample_0 = 1
                     #   weights_0_for_IS_estimator_ref1.append(prior_sample_0/float(instrumental_sample_0))
                        #print weights_0_for_IS_estimator_ref1[-1]
                      #  N_ref1_all +=1
                        #print 'at no GW but if d ', N_ref1_all
                
                #else:
                 #   prior_sample_0 = 1#prior_M1(float(sample_ref1_single[0]))*prior_a(float(sample_ref1_single[1]))*prior_q(float(sample_ref1_single[2])) 
                  #  #instrumental_sample_0 = (multivariate_normal.pdf(sample_ref1_single,gaus_mean_for_ref1[i],cov))
                   # instrumental_sample_0 = 1
                    #weights_0_for_IS_estimator_ref1.append(prior_sample_0/float(instrumental_sample_0))
                    #print weights_0_for_IS_estimator_ref1[-1]
                   # N_ref1_all +=1
                    #print 'at no GW', N_ref1_all
                #N_ref1_all  += N_ini_ones_choice 

      #  print 'at final', N_ref1_all, 'inside', len(weights_1), 'outside', count_outside_parameter_space
        #print 'len weights 1', len(weights_1_for_IS_estimator_ref1),weights_1_for_IS_estimator_ref1
        #print 'len weights 0', len(weights_0_for_IS_estimator_ref1),weights_0_for_IS_estimator_ref1
        #print 'this should give the same', len(samples_all_ref1)
        ISestimator = ImportanceSampling_estimator_ref1(N_ref1_all,weights_1_for_IS_estimator_ref1) 
        #ISestimator2 = ImportanceSampling_estimator_ref1((N_ref1_all+count_outside_parameter_space),weights_1_for_IS_estimator_ref1)
        print ' the IS estimator = ', ISestimator
     #   print 'the IS2 (with rejected) estimator = ',ISestimator2
       # print 'number of merging BHs = ', len(M1_ref1_1)

        #print 'the IS estimator times N = ', ISestimator*N_ref1_all 
        estimator_error = estimator_error -0.2
            #print estimator_error
    print 'number of rejected samples = ', count_outside_parameter_space
    print 'total GWs in ref1 =', len(M1_ref1_1), 'total nr of samples = ', N_ref1_all





N_ini_ones_choice = 1000000
N_ini_all,M1_ini_1,weights_ini_1,gaus_mean_for_ref1 =  initial_sampling(N_ini_ones_choice)

print 'N_ini_all = ', N_ini_all
print len(weights_ini_1)
#print M1_ini_1
print 'IS estimator = ', (1./N_ini_all)*sum(weights_ini_1)

#print gaus_mean_for_ref1
#refined_1_sampling(N_ini_ones_choice,gaus_mean_for_ref1,N_ini_all)
