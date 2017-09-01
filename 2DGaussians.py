# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:26:19 2017

@author: Floor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D 

size = 100
sigma_x = 6.
sigma_y = 2.

x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)

x, y = np.meshgrid(x, y)
z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
     + y**2/(2*sigma_y**2))))

plt.contourf(x, y, z, cmap='Blues')
plt.colorbar()
plt.show()

 #mostly true for stratified initial sampling
cov =  [[0.1**2, 0], [0, 9]] 
samples_ref1 = []
mean0 = [0,0]

samples2D = np.random.multivariate_normal(mean0, cov, 1000) 

xx = samples2D[:,0]
yy = samples2D[:,1]
zz = multivariate_normal.pdf(samples2D,mean0,cov)

#    
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(xx,yy,zz) 
plt.show()  
##################
#samples2Duni = np.random.uniform(mean0, cov, 1000) 
#
#xx = samples2D[:,0]
#yy = samples2D[:,1]
#zz = multivariate_normal.pdf(samples2D,mean0,cov)
#
##    
#fig = plt.figure() 
#ax = fig.add_subplot(111, projection='3d') 
#ax.scatter(xx,yy,zz) 
#plt.show()  
