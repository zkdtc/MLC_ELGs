# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight - learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
### Read Data #####
####################################
from astropy.table import Table
import matplotlib.pyplot as plt
import os
import pickle

n_input_pars=8
n_output_pars=4
input_file='data_matched_elg_step2.csv'
net_name0='model_rf_0.sav'
net_name1='model_rf_1.sav'
net_name2='model_rf_2.sav'
net_name3='model_rf_3.sav'
net_name4='model_rf_4.sav'
net_name5='model_rf_5.sav'

# load the model from disk
model0 = pickle.load(open(net_name0, 'rb'))
model1 = pickle.load(open(net_name1, 'rb'))
model2 = pickle.load(open(net_name2, 'rb'))
model3 = pickle.load(open(net_name3, 'rb'))
model4 = pickle.load(open(net_name4, 'rb'))
model5 = pickle.load(open(net_name5, 'rb'))

data = Table.read(input_file,format='ascii.csv',header_start=0,data_start=1)
ind1=np.where(np.array(data['o3']/data['o3_err']) >3)
ind2=np.where(np.array((data['o21']+data['o22'])/np.sqrt(data['o21_err']**2+data['o22_err']**2)) >3)
ind3=np.where(np.array(data['hb']/data['hb_err']) >3)
ind4=np.where(np.array(data['sigma_o3']/data['sigma_o3_err']) >3)
ind5=np.where(np.array(data['mag_u']) >10)
ind6=np.where(np.array(data['mag_u']) <40.)
ind7=np.where(np.array(data['mag_i']) >5.)
ind8=np.where(np.array(data['mag_i']) <40.)

ind_g=np.where(np.array(data['flux_g']) >0.)
ind_r=np.where(np.array(data['flux_r']) >0.)
ind_z=np.where(np.array(data['flux_z']) >0.)
data['mag_g'][ind_g]=22.5-2.5*np.log10(data['flux_g'][ind_g])
data['mag_r'][ind_r]=22.5-2.5*np.log10(data['flux_r'][ind_r])
data['mag_z'][ind_z]=22.5-2.5*np.log10(data['flux_z'][ind_z])
ind9=np.where(np.array(data['mag_g']) >10)
ind10=np.where(np.array(data['mag_g']) <40.)
ind11=np.where(np.array(data['mag_r']) >5.)
ind12=np.where(np.array(data['mag_r']) <40.)
ind13=np.where(np.array(data['mag_z']) >5.)
ind14=np.where(np.array(data['mag_z']) <40.)

ind=np.array(list(set(ind1[0]) & set(ind2[0]) & set(ind3[0]) & set(ind4[0]) & set(ind5[0]) & set(ind6[0])  & set(ind7[0]) & set(ind8[0]) & set(ind9[0]) & set(ind10[0]) & set(ind11[0]) & set(ind12[0]) & set(ind13[0]) & set(ind14[0])  ))
n_source=len(ind)
maggie_u=10**((22.5-data['mag_u'][ind])/2.5)
maggie_g=10**((22.5-data['mag_g'][ind])/2.5)
maggie_r=10**((22.5-data['mag_r'][ind])/2.5)
maggie_i=10**((22.5-data['mag_i'][ind])/2.5)
maggie_z=10**((22.5-data['mag_z'][ind])/2.5)

import kcorrect, numpy
kcorrect.load_templates()
kcorrect.load_filters()

for i in range(n_source):
    print(i)
    a=[data['z'][ind[i]], maggie_u[i],maggie_g[i],maggie_r[i],maggie_i[i],maggie_z[i], 6.216309e+16, 3.454767e+17, 1.827409e+17, 1.080889e+16, 3163927000000000.0]
    c = kcorrect.fit_coeffs(a)
    m = kcorrect.reconstruct_maggies(c)
    maggie_u[i]=m[1]
    maggie_g[i]=m[2]
    maggie_r[i]=m[3]
    maggie_i[i]=m[4]
    maggie_z[i]=m[5]



type_arr=np.zeros(len(ind))
type_arr=type_arr-999
O2_index=np.log10((data['o21'][ind]+data['o22'][ind])/data['hb'][ind])

O3_index=np.log10(data['o3'][ind]/data['hb'][ind])
sigma_o3=np.log10(np.sqrt(data['sigma_o3'][ind]**2)) #-55**2))
sigma_star=np.log10(data['VDISP'][ind])
indt=np.where(sigma_star ==0)
sigma_star[indt]=10.


indt=np.where(maggie_u <=0)
maggie_u[indt]=0.001
indt=np.where(maggie_g <=0)
maggie_g[indt]=0.001
indt=np.where(maggie_r <=0)
maggie_r[indt]=0.001
indt=np.where(maggie_i <=0)
maggie_i[indt]=0.001
indt=np.where(maggie_z <=0)
maggie_z[indt]=0.001

mag_u=22.5-2.5*np.log10(maggie_u)
mag_g=22.5-2.5*np.log10(maggie_g)
mag_r=22.5-2.5*np.log10(maggie_r)
mag_i=22.5-2.5*np.log10(maggie_i)
mag_z=22.5-2.5*np.log10(maggie_z)

#u_g=data['mag_u'][ind]-data['mag_g'][ind]
#g_r=data['mag_g'][ind]-data['mag_r'][ind]
#r_i=data['mag_r'][ind]-data['mag_i'][ind]
#i_z=data['mag_i'][ind]-data['mag_z'][ind]
u_g=mag_u-mag_g
g_r=mag_g-mag_r
r_i=mag_r-mag_i
i_z=mag_i-mag_z

O3_index[O3_index == -np.inf] = 0
O2_index[O2_index == -np.inf] = 0
sigma_star[sigma_star == -np.inf] = 0
sigma_o3[sigma_o3 == -np.inf] = 0
u_g[u_g == -np.inf] = 0
g_r[g_r == -np.inf] = 0
r_i[r_i == -np.inf] = 0
i_z[i_z == -np.inf] = 0
O3_index[O3_index == np.inf] = 0
O2_index[O2_index == np.inf] = 0
sigma_star[sigma_star == np.inf] = 0
sigma_o3[sigma_o3 == np.inf] = 0
u_g[u_g == np.inf] = 0
g_r[g_r == np.inf] = 0
r_i[r_i == np.inf] = 0
i_z[i_z == np.inf] = 0

O3_index=np.nan_to_num(O3_index)
O2_index=np.nan_to_num(O2_index)
sigma_star=np.nan_to_num(sigma_star)
sigma_o3=np.nan_to_num(sigma_o3)
u_g=np.nan_to_num(u_g)
g_r=np.nan_to_num(g_r)
r_i=np.nan_to_num(r_i)
i_z=np.nan_to_num(i_z)

markersize=1
X_test=[]
for i in range(n_source):
    input_this=[float(O3_index[i]),float(O2_index[i]),float(sigma_star[i]),float(sigma_o3[i]) ,float(u_g[i]),float(g_r[i]),float(r_i[i]),float(i_z[i]) ]
    X_test.append(input_this)
type_arr_out0=np.array(model0.predict(X_test))
model0=0
type_arr_out1=np.array(model1.predict(X_test))
model1=0
type_arr_out2=np.array(model2.predict(X_test))
model2=0
#type_arr_out3=np.array(model3.predict(X_test))
#model3=0
#type_arr_out4=np.array(model4.predict(X_test))
#model4=0
#type_arr_out5=np.array(model5.predict(X_test))
#model5=0
type_arr_out=[]
for i in range(n_source):
    type_this=np.array([float(type_arr_out0[i]),float(type_arr_out1[i]),float(type_arr_out2[i])]) #,float(type_arr_out3[i]),float(type_arr_out4[i]),float(type_arr_out5[i]) ])
    counts = np.bincount(type_this.astype(int))
    type_arr_out.append(np.argmax(counts))
type_arr_out=np.array(type_arr_out)

ind_new_sf=np.where(type_arr_out ==1)
ind_new_comp=np.where(type_arr_out ==2)
ind_new_AGN=np.where(type_arr_out ==3)
ind_new_liner=np.where(type_arr_out ==4)
xx=np.array([0,1,2,3])
yy=-2.*xx+4.2

plt.subplot(2, 2, 1)
plt.plot(sigma_o3[ind_new_sf],O3_index[ind_new_sf],'b+',markersize=markersize)
plt.plot(sigma_o3[ind_new_comp],O3_index[ind_new_comp],'g+',markersize=markersize)
plt.plot(sigma_o3[ind_new_AGN],O3_index[ind_new_AGN],'r+',markersize=markersize)
plt.plot(sigma_o3[ind_new_liner],O3_index[ind_new_liner],'y+',markersize=markersize)
plt.plot(xx,yy,'b')
plt.xlabel(r'$\sigma$([OIII])')
plt.ylabel(r'Log [OIII]/H$\beta$')
plt.xlim(1.5,2.8)
plt.ylim(-1,1.5)

plt.subplot(2, 2, 2)
plt.plot(sigma_o3[ind_new_sf],O3_index[ind_new_sf],'b+',markersize=markersize)
plt.plot(sigma_o3[ind_new_liner],O3_index[ind_new_liner],'y+',markersize=markersize)
plt.plot(xx,yy,'b')
plt.xlabel(r'$\sigma$([OIII])')
plt.ylabel(r'Log [OIII]/H$\beta$')
plt.xlim(1.5,2.8)
plt.ylim(-1,1.5)

plt.subplot(2, 2, 3)
plt.plot(sigma_o3[ind_new_comp],O3_index[ind_new_comp],'g+',markersize=markersize)
plt.plot(xx,yy,'b')
plt.xlabel(r'$\sigma$([OIII])')
plt.ylabel(r'Log [OIII]/H$\beta$')
plt.xlim(1.5,2.8)
plt.ylim(-1,1.5)

plt.subplot(2, 2, 4)
plt.plot(sigma_o3[ind_new_AGN],O3_index[ind_new_AGN],'r+',markersize=markersize)
plt.plot(xx,yy,'b')
plt.xlabel(r'$\sigma$([OIII])')
plt.ylabel(r'Log [OIII]/H$\beta$')
plt.xlim(1.5,2.8)
plt.ylim(-1,1.5)

plt.show()

os.system('rm catalog*.csv')

f=open('catalog_all.csv','w')
from astropy.table import Table
table=Table([[],[],[],[]],names=['MJD','PLATE','FIBERID','TYPE'],dtype=['S8','S8','S8','S8'])
for i in range(len(ind)):
    indind=ind[i]
    outline=str(data['MJD'][indind])+','+str(data['PLATE'][indind])+','+str(data['FIBERID'][indind])+','+str(type_arr_out[i])+'\n'
    table.add_row([str(data['MJD'][indind]),str(data['PLATE'][indind]),str(data['FIBERID'][indind]),str(type_arr_out[i])])
    f.write(outline)
f.close()
table.write('catalog_all.fits', format='fits',overwrite=True)


ind_this=ind_new_sf
f=open('catalog_sf.csv','w')
for i in range(len(ind_this[0])):
    indind=ind[ind_this[0][i]]
    outline=str(data['MJD'][indind])+','+str(data['PLATE'][indind])+','+str(data['FIBERID'][indind])+'\n'
    f.write(outline)
f.close()

ind_this=ind_new_comp
f=open('catalog_comp.csv','w')
for i in range(len(ind_this[0])):
    indind=ind[ind_this[0][i]]
    outline=str(data['MJD'][indind])+','+str(data['PLATE'][indind])+','+str(data['FIBERID'][indind])+'\n'
    f.write(outline)
f.close()

ind_this=ind_new_AGN
f=open('catalog_AGN.csv','w')
for i in range(len(ind_this[0])):
    indind=ind[ind_this[0][i]]
    outline=str(data['MJD'][indind])+','+str(data['PLATE'][indind])+','+str(data['FIBERID'][indind])+'\n'
    f.write(outline)
f.close()

ind_this=ind_new_liner
f=open('catalog_liner.csv','w')
for i in range(len(ind_this[0])):
    indind=ind[ind_this[0][i]]
    outline=str(data['MJD'][indind])+','+str(data['PLATE'][indind])+','+str(data['FIBERID'][indind])+'\n'
    f.write(outline)
f.close()


import pdb;pdb.set_trace()
