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
from sklearn import svm
n_input_pars=8
n_output_pars=4
from sklearn.neighbors import KNeighborsClassifier

###################################
### Read Data #####
####################################
from astropy.table import Table
import matplotlib.pyplot as plt

data = Table.read('data_matched_step2.csv',format='ascii.csv',header_start=0,data_start=1)
ind1=np.where(np.array(data['o3']/data['o3_err']) >3)
ind2=np.where(np.array((data['o21']+data['o22'])/np.sqrt(data['o21_err']**2+data['o22_err']**2)) >3)
ind3=np.where(np.array(data['hb']/data['hb_err']) >3)
ind4=np.where(np.array(data['ha']/data['ha_err']) >3)
ind5=np.where(np.array(data['s21']/data['s21_err']) >3)
ind6=np.where(np.array(data['n2']/data['n2_err']) >3)
ind7=np.where(np.array(data['sigma_o3']/data['sigma_o3_err']) >3)
ind8=np.where(np.array(data['VDISP'])>0.)
ind9=np.where(np.array(data['mag_u'])>10.)
ind10=np.where(np.array(data['flux_w1'])>0.)



ind_g=np.where(np.array(data['flux_g']) >0.)
ind_r=np.where(np.array(data['flux_r']) >0.)
ind_z=np.where(np.array(data['flux_z']) >0.)
data['mag_g'][ind_g]=22.5-2.5*np.log10(data['flux_g'][ind_g])
data['mag_r'][ind_r]=22.5-2.5*np.log10(data['flux_r'][ind_r])
data['mag_z'][ind_z]=22.5-2.5*np.log10(data['flux_z'][ind_z])



ind=np.array(list(set(ind1[0]) & set(ind1[0]) &set(ind2[0]) & set(ind3[0]) & set(ind4[0]) & set(ind5[0]) & set(ind6[0]) & set(ind7[0]) & set(ind8[0])))# & set(ind9[0])& set(ind10[0]) ))
n_source=len(ind)
n_split=int(n_source*0.7)
ind_train=ind[0:n_split]
ind_test=ind[n_split:]
n_train=len(ind_train)
n_test=len(ind_test)
type_arr=np.zeros(len(ind))
type_arr=type_arr-999
z=np.array(data['z'][ind])
O2_index=np.log10((data['o21'][ind]+data['o22'][ind])/data['hb'][ind])
O3_index=np.log10(data['o3'][ind]/data['hb'][ind])
N2_index=np.log10(data['n2'][ind]/data['ha'][ind])
S2_index=np.log10((data['s21'][ind]+data['s22'][ind])/data['ha'][ind])
sigma_o3=np.log10(data['sigma_o3'][ind])
sigma_star=np.log10(data['VDISP'][ind])
u_g=data['mag_u'][ind]-data['mag_g'][ind]
g_r=data['mag_g'][ind]-data['mag_r'][ind]
r_i=data['mag_r'][ind]-data['mag_i'][ind]
i_z=data['mag_i'][ind]-data['mag_z'][ind]
z_w1=data['mag_z'][ind]-(22.5-2.5*np.log10(data['flux_w1'][ind]))
ind_sf1=np.where(O3_index <= (0.61/(N2_index-0.05)+1.3))
ind_sf2=np.where( N2_index < 0.)
ind_sf=np.array(list(set(ind_sf1[0]) & set(ind_sf2[0])))

ind_AGN1=np.where(O3_index > (0.61/(N2_index-0.47)+1.19))
ind_AGN2=np.where(N2_index >= 0.)
ind_AGN3=np.where(O3_index > 1.89*S2_index+0.76)
ind_AGN=np.array(list((set(ind_AGN1[0]) | set(ind_AGN2[0])) & set(ind_AGN3[0])))


ind_liner1=np.where(O3_index > (0.61/(N2_index-0.47)+1.19))
ind_liner2=np.where(O3_index <= 1.89*S2_index+0.76)
ind_liner=np.array(list(set(ind_liner1[0]) & set(ind_liner2[0])))


ind_comp1=np.where(O3_index < (0.61/(N2_index-0.47)+1.19))
ind_comp2=np.where(O3_index > (0.61/(N2_index-0.05)+1.3))
ind_comp=np.array(list(set(ind_comp1[0]) & set(ind_comp2[0])))

type_arr[ind_sf]=1
type_arr[ind_comp]=2
type_arr[ind_AGN]=3
type_arr[ind_liner]=4



type_arr_train=np.zeros(len(ind_train))
type_arr_train=type_arr_train-999

O2_index_train=np.log10((data['o21'][ind_train]+data['o22'][ind_train])/data['hb'][ind_train])
O3_index_train=np.log10(data['o3'][ind_train]/data['hb'][ind_train])
N2_index_train=np.log10(data['n2'][ind_train]/data['ha'][ind_train])
S2_index_train=np.log10((data['s21'][ind_train]+data['s22'][ind_train])/data['ha'][ind_train])
sigma_o3_train=np.log10(data['sigma_o3'][ind_train])
sigma_star_train=np.log10(data['VDISP'][ind_train])
u_g_train=data['mag_u'][ind_train]-data['mag_g'][ind_train]
g_r_train=data['mag_g'][ind_train]-data['mag_r'][ind_train]
r_i_train=data['mag_r'][ind_train]-data['mag_i'][ind_train]
i_z_train=data['mag_i'][ind_train]-data['mag_z'][ind_train]
z_w1_train=data['mag_z'][ind_train]-(22.5-np.log10(data['flux_w1'][ind_train]))


ind_sf1_train=np.where(O3_index_train <= (0.61/(N2_index_train-0.05)+1.3))
ind_sf2_train=np.where( N2_index_train < 0.)
ind_sf_train=np.array(list(set(ind_sf1_train[0]) & set(ind_sf2_train[0])))

ind_AGN1_train=np.where(O3_index_train > (0.61/(N2_index_train-0.47)+1.19))
ind_AGN2_train=np.where(N2_index_train >= 0.)
ind_AGN3_train=np.where(O3_index_train > 1.89*S2_index_train+0.76)
ind_AGN_train=np.array(list(set(ind_AGN1_train[0]) | set(ind_AGN2_train[0]) & set(ind_AGN3_train[0])))


ind_liner1_train=np.where(O3_index_train > (0.61/(N2_index_train-0.47)+1.19)) 
ind_liner2_train=np.where(O3_index_train <= 1.89*S2_index_train+0.76)
ind_liner_train=np.array(list(set(ind_liner1_train[0]) & set(ind_liner2_train[0])))


ind_comp1_train=np.where(O3_index_train < (0.61/(N2_index_train-0.47)+1.19))
ind_comp2_train=np.where(O3_index_train > (0.61/(N2_index_train-0.05)+1.3))
ind_comp_train=np.array(list(set(ind_comp1_train[0]) & set(ind_comp2_train[0])))


type_arr_train[ind_sf_train]=1
type_arr_train[ind_comp_train]=2
type_arr_train[ind_AGN_train]=3
type_arr_train[ind_liner_train]=4

type_arr_test=np.zeros(len(ind_test))
type_arr_test=type_arr_test-999

O2_index_test=np.log10((data['o21'][ind_test]+data['o22'][ind_test])/data['hb'][ind_test])
O3_index_test=np.log10(data['o3'][ind_test]/data['hb'][ind_test])
N2_index_test=np.log10(data['n2'][ind_test]/data['ha'][ind_test])
S2_index_test=np.log10((data['s21'][ind_test]+data['s22'][ind_test])/data['ha'][ind_test])
sigma_o3_test=np.log10(data['sigma_o3'][ind_test])
sigma_star_test=np.log10(data['VDISP'][ind_test])
u_g_test=data['mag_u'][ind_test]-data['mag_g'][ind_test]
g_r_test=data['mag_g'][ind_test]-data['mag_r'][ind_test]
r_i_test=data['mag_r'][ind_test]-data['mag_i'][ind_test]
i_z_test=data['mag_i'][ind_test]-data['mag_z'][ind_test]
z_w1_test=data['mag_z'][ind_test]-(22.5-np.log10(data['flux_w1'][ind_test]))

ind_sf1_test=np.where(O3_index_test <= (0.61/(N2_index_test-0.05)+1.3))
ind_sf2_test=np.where( N2_index_test < 0.)
ind_sf_test=np.array(list(set(ind_sf1_test[0]) & set(ind_sf2_test[0])))

ind_AGN1_test=np.where(O3_index_test > (0.61/(N2_index_test-0.47)+1.19))
ind_AGN2_test=np.where(N2_index_test >= 0.)
ind_AGN3_test=np.where(O3_index_test > 1.89*S2_index_test+0.76)
ind_AGN_test=np.array(list(set(ind_AGN1_test[0]) | set(ind_AGN2_test[0]) & set(ind_AGN3_test[0])))


ind_liner1_test=np.where(O3_index_test > (0.61/(N2_index_test-0.47)+1.19))
ind_liner2_test=np.where(O3_index_test <= 1.89*S2_index_test+0.76)
ind_liner_test=np.array(list(set(ind_liner1_test[0]) & set(ind_liner2_test[0])))


ind_comp1_test=np.where(O3_index_test < (0.61/(N2_index_test-0.47)+1.19))
ind_comp2_test=np.where(O3_index_test > (0.61/(N2_index_test-0.05)+1.3))
ind_comp_test=np.array(list(set(ind_comp1_test[0]) & set(ind_comp2_test[0])))

type_arr_test[ind_sf_test]=1
type_arr_test[ind_comp_test]=2
type_arr_test[ind_AGN_test]=3
type_arr_test[ind_liner_test]=4




markersize=1


#######################################################################
### Learning Code starts here #####
######################################

mark_points=(1+np.arange(50))*0.02*float(6000)
mark_points=mark_points.astype(int)
n_mark_points=len(mark_points)
accuracy_sf=np.zeros(n_mark_points)
accuracy_comp=np.zeros(n_mark_points)
accuracy_liner=np.zeros(n_mark_points)
accuracy_AGN=np.zeros(n_mark_points)
accuracy_overall=np.zeros(n_mark_points)
completeness_sf=np.zeros(n_mark_points)
completeness_comp=np.zeros(n_mark_points)
completeness_liner=np.zeros(n_mark_points)
completeness_AGN=np.zeros(n_mark_points)

median_sf=[]
median_comp=[]
median_AGN=[]
median_liner=[]

var_arr=['O3_index','O2_index','sigma_star','sigma_o3','u_g','g_r','r_i','i_z']
for var in var_arr:
    cmd='t=(np.median('+var+'[ind_sf])-np.percentile('+var+',5))/(np.percentile('+var+',95)-np.percentile('+var+',5))'
    exec(cmd)
    median_sf.append(float(t))
for var in var_arr:
    cmd='t=(np.median('+var+'[ind_comp])-np.percentile('+var+',5))/(np.percentile('+var+',95)-np.percentile('+var+',5))'
    exec(cmd)
    median_comp.append(float(t))
for var in var_arr:
    cmd='t=(np.median('+var+'[ind_AGN])-np.percentile('+var+',5))/(np.percentile('+var+',95)-np.percentile('+var+',5))'
    exec(cmd)
    median_AGN.append(float(t))
for var in var_arr:
    cmd='t=(np.median('+var+'[ind_liner])-np.percentile('+var+',5))/(np.percentile('+var+',95)-np.percentile('+var+',5))'
    exec(cmd)
    median_liner.append(float(t))

axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect=1, cmap='coolwarm', interpolation=None)

import matplotlib as mpl

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
ind=0
image_arr=[median_sf,median_comp,median_AGN,median_liner]
title_arr=['SFGs','Composites','AGNs','LINERs']
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_arr[ind])
    im = ax.imshow([image_arr[ind]], vmin=0, vmax=1,cmap='coolwarm')
    ind=ind+1
y_anchor=1
interval=1.05
plt.text(-0.5,y_anchor,r'Log [OIII]/H$\beta$',fontsize=7)
plt.text(0.5,y_anchor,r'Log [OII]/H$\beta$',fontsize=7)
plt.text(1.6,y_anchor,r'Log $\sigma$([OIII])',fontsize=7)
plt.text(2.8,y_anchor,r'Log $\sigma$*',fontsize=7)
plt.text(3.8,y_anchor,'u-g')
plt.text(3.8+interval*1,y_anchor,'g-r')
plt.text(3.8+interval*2,y_anchor,'r-i')
plt.text(3.8+interval*3,y_anchor,'i-z')


cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(im, cax=cax, **kw)

plt.show()

"""
fig = plt.figure()
ax1 = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax1.imshow([median_sf], **barprops)

ax2 = fig.add_axes([0.3, 0.25, 0.6, 0.25], **axprops)
ax2.imshow([median_comp], **barprops)

ax3 = fig.add_axes([0.3, 0.4, 0.6, 0.4], **axprops)
ax3.imshow([median_AGN], **barprops)

ax4 = fig.add_axes([0.3, 0.55, 0.6, 0.55], **axprops)
ax4.imshow([median_liner], **barprops)

ax5 = fig.add_axes([0.3, 0.4,0.6, 0.4], **axprops)
im=ax5.imshow([[0,0,0]], alpha=1,**barprops)

cbarlabel='normalized value'
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
"""

X=[]
Y=[]
n_round=10000
for i in range(n_round):
    if i%4==0:
        ind_select=ind_sf_train
        type_this=1
    elif i%4==1:
        ind_select=ind_comp_train
        type_this=2
    elif i%4==2:
        ind_select=ind_AGN_train
        type_this=3
    elif i%4==3:
        ind_select=ind_liner_train
        type_this=4

    select=np.random.randint(0,len(ind_select)-1)

    input_this=[float(O3_index[ind_select[select]]),float(O2_index[ind_select[select]]),float(sigma_star[ind_select[select]]),float(sigma_o3[ind_select[select]]),float(u_g[ind_select[select]]),float(g_r[ind_select[select]]),float(r_i[ind_select[select]]),float(i_z[ind_select[select]])]
    if type_this<0:
        pass
    else:
        X.append(input_this)
        Y.append(type_this)

n_neighbour_arr=np.arange(100)+1
n_try=len(n_neighbour_arr)
accuracy_sf_knn=np.zeros(n_try)
accuracy_comp_knn=np.zeros(n_try)
accuracy_liner_knn=np.zeros(n_try)
accuracy_AGN_knn=np.zeros(n_try)
accuracy_overall_knn=np.zeros(n_try)
completeness_sf_knn=np.zeros(n_try)
completeness_comp_knn=np.zeros(n_try)
completeness_liner_knn=np.zeros(n_try)
completeness_AGN_knn=np.zeros(n_try)

for j in range(n_try):
    n_neighbour=n_neighbour_arr[j]
    clf = KNeighborsClassifier(n_neighbors=n_neighbour)
    clf.fit(X, Y)

    X_test=[]
    for i in range(n_test):
        input_this=[float(O3_index_test[i]),float(O2_index_test[i]),float(sigma_star_test[i]),float(sigma_o3_test[i]),float(u_g_test[i]),float(g_r_test[i]),float(r_i_test[i]),float(i_z_test[i]) ]
        X_test.append(input_this)

    type_arr_out=np.array(clf.predict(X_test))
    ind_new_sf=np.where(type_arr_out ==1)
    ind_new_comp=np.where(type_arr_out ==2)
    ind_new_AGN=np.where(type_arr_out ==3)
    ind_new_liner=np.where(type_arr_out ==4)

    test1=type_arr_test[ind_new_sf]
    ind1=np.where(test1==1)
    if len(ind_new_sf[0])==0:
        accuracy_sf_knn[j]=0.
        completeness_sf_knn[j]=0.
        print(len(ind1[0]),len(ind_new_sf[0]),accuracy_sf_knn[j])
    else:
        accuracy_sf_knn[j]=float(len(ind1[0]))/float(len(ind_new_sf[0]))
        completeness_sf_knn[j]=float(len(ind1[0]))/float(len(ind_sf_test))
        print(len(ind1[0]),len(ind_new_sf[0]),len(ind_sf_test),accuracy_sf_knn[j],completeness_sf_knn[j])

    test2=type_arr_test[ind_new_comp]
    ind2=np.where(test2==2)
    if len(ind_new_comp[0])==0:
        accuracy_comp_knn[j]=0.
        completeness_comp_knn[j]=0.
        print(len(ind2[0]),len(ind_new_comp[0]),accuracy_comp_knn[j])
    else:
        accuracy_comp_knn[j]=float(len(ind2[0]))/float(len(ind_new_comp[0]))
        completeness_comp_knn[j]=float(len(ind2[0]))/float(len(ind_comp_test))
        print(len(ind2[0]),len(ind_new_comp[0]),len(ind_comp_test),accuracy_comp_knn[j],completeness_comp_knn[j])

    test3=type_arr_test[ind_new_AGN]
    ind3=np.where(test3==3)
    if len(ind_new_AGN[0])==0:
        accuracy_AGN_knn[j]=0.
        completeness_AGN_knn[j]=0.
        print(len(ind3[0]),len(ind_new_AGN[0]),accuracy_AGN_knn[j])
    else:
        accuracy_AGN_knn[j]=float(len(ind3[0]))/float(len(ind_new_AGN[0]))
        completeness_AGN_knn[j]=float(len(ind3[0]))/float(len(ind_AGN_test))
        print(len(ind3[0]),len(ind_new_AGN[0]),len(ind_AGN_test),accuracy_AGN_knn[j],completeness_AGN_knn[j])

    test4=type_arr_test[ind_new_liner]
    ind4=np.where(test4==4)
    if len(ind_new_liner[0])==0:
        accuracy_liner_knn[j]=0.
        completeness_liner_knn[j]=0.
        print(len(ind4[0]),len(ind_new_liner[0]),accuracy_liner_knn[j])
    else:
        accuracy_liner_knn[j]=float(len(ind4[0]))/float(len(ind_new_liner[0]))
        completeness_liner_knn[j]=float(len(ind4[0]))/float(len(ind_liner_test))
        print(len(ind4[0]),len(ind_new_liner[0]),len(ind_liner_test),accuracy_liner_knn[j],completeness_liner_knn[j])

accuracy_overall_knn=(accuracy_AGN_knn+accuracy_sf_knn+accuracy_comp_knn+accuracy_liner_knn)/4.




fig=plt.figure(num=0,figsize=(6,5))
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.plot(n_neighbour_arr,accuracy_overall_knn,alpha=0.5)

plt.axis([0,n_neighbour_arr[-1], 0, 1])
plt.xlabel('N Neighbors')
plt.ylabel('Overall Classification Accuracy')
plt.title('KNN N Neighbors Choice')
plt.legend(loc='lower right')

ind = np.unravel_index(np.argmax(accuracy_overall_knn, axis=None), accuracy_overall_knn.shape)
best_n_neighbour=n_neighbour_arr[ind]
print('Best n_neighbour',best_n_neighbour)
fig.tight_layout()
plt.savefig('knn_choose.pdf', format='pdf', dpi=1000)
plt.close()
#plt.show()


fig=plt.figure(num=1,figsize=(14,6))
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

markersize=4
alpha=0.5
plt.subplot(1, 2, 1)
plt.plot(N2_index[ind_sf],O3_index[ind_sf],'b+',markersize=markersize,alpha=alpha,label='SFGs')
plt.plot(N2_index[ind_comp],O3_index[ind_comp],'g+',markersize=markersize,alpha=alpha,label='Composites')
plt.plot(N2_index[ind_AGN],O3_index[ind_AGN],'r+',markersize=markersize,alpha=alpha,label='AGNs')
plt.plot(N2_index[ind_liner],O3_index[ind_liner],'+',color='orange',markersize=markersize,alpha=alpha,label='LINERs')

plt.xlabel(r'Log [NII]/H$\alpha$')
plt.ylabel(r'Log [OIII]/H$\beta$')
plt.legend(loc='lower left')

plt.subplot(1, 2, 2)
plt.xlabel(r'Log [SII]/H$\alpha$')
plt.ylabel(r'Log [OIII]/H$\beta$')

plt.plot(S2_index[ind_sf],O3_index[ind_sf],'b+',markersize=markersize,alpha=alpha)
plt.plot(S2_index[ind_comp],O3_index[ind_comp],'g+',markersize=markersize,alpha=alpha)
plt.plot(S2_index[ind_AGN],O3_index[ind_AGN],'r+',markersize=markersize,alpha=alpha)
plt.plot(S2_index[ind_liner],O3_index[ind_liner],'+',color='orange',markersize=markersize,alpha=alpha)

fig.tight_layout()
plt.savefig('bpt.pdf', format='pdf', dpi=1000)
plt.close()

import pdb; pdb.set_trace()

for j in range(n_mark_points):
    n_round=mark_points
    X=[]
    Y=[]
    n_round=mark_points[j]
    for i in range(n_round):
        if i%4==0:
            ind_select=ind_sf_train
            type_this=1
        elif i%4==1:
            ind_select=ind_comp_train
            type_this=2
        elif i%4==2:
            ind_select=ind_AGN_train
            type_this=3
        elif i%4==3:
            ind_select=ind_liner_train
            type_this=4

        select=np.random.randint(0,len(ind_select)-1)
    
        input_this=[float(O3_index[ind_select[select]]),float(O2_index[ind_select[select]]),float(sigma_star[ind_select[select]]),float(sigma_o3[ind_select[select]]),float(u_g[ind_select[select]]),float(g_r[ind_select[select]]),float(r_i[ind_select[select]]),float(i_z[ind_select[select]])]
        if type_this<0:
            pass
        else:
            X.append(input_this)
            Y.append(type_this)

    
# Mark Point Check
        
    print(str(i)+' sources used')
    #clf = svm.LinearSVC() 
    #clf = svm.SVC( decision_function_shape='ovo')
    title='SVM Classification'
    clf = KNeighborsClassifier(n_neighbors=int(best_n_neighbour))
    title='KNN Classification'
    clf.fit(X, Y) 

    X_test=[]
    for i in range(n_test):
        input_this=[float(O3_index_test[i]),float(O2_index_test[i]),float(sigma_star_test[i]),float(sigma_o3_test[i]),float(u_g_test[i]),float(g_r_test[i]),float(r_i_test[i]),float(i_z_test[i]) ]
        X_test.append(input_this)
     
    type_arr_out=np.array(clf.predict(X_test))
    ind_new_sf=np.where(type_arr_out ==1)
    ind_new_comp=np.where(type_arr_out ==2)
    ind_new_AGN=np.where(type_arr_out ==3)
    ind_new_liner=np.where(type_arr_out ==4)

    test1=type_arr_test[ind_new_sf]
    ind1=np.where(test1==1)
    if len(ind_new_sf[0])==0:
        accuracy_sf[j]=0.
        completeness_sf[j]=0.
        print(len(ind1[0]),len(ind_new_sf[0]),accuracy_sf[j])
    else:
        accuracy_sf[j]=float(len(ind1[0]))/float(len(ind_new_sf[0]))
        completeness_sf[j]=float(len(ind1[0]))/float(len(ind_sf_test))
        print(len(ind1[0]),len(ind_new_sf[0]),len(ind_sf_test),accuracy_sf[j],completeness_sf[j])

    test2=type_arr_test[ind_new_comp]
    ind2=np.where(test2==2)
    if len(ind_new_comp[0])==0:
        accuracy_comp[j]=0.
        completeness_comp[j]=0.
        print(len(ind2[0]),len(ind_new_comp[0]),accuracy_comp[j])
    else:
        accuracy_comp[j]=float(len(ind2[0]))/float(len(ind_new_comp[0]))
        completeness_comp[j]=float(len(ind2[0]))/float(len(ind_comp_test))
        print(len(ind2[0]),len(ind_new_comp[0]),len(ind_comp_test),accuracy_comp[j],completeness_comp[j])

    test3=type_arr_test[ind_new_AGN]
    ind3=np.where(test3==3)
    if len(ind_new_AGN[0])==0:
        accuracy_AGN[j]=0.
        completeness_sf[j]=0.
        print(len(ind3[0]),len(ind_new_AGN[0]),accuracy_AGN[j])
    else:
        accuracy_AGN[j]=float(len(ind3[0]))/float(len(ind_new_AGN[0]))
        completeness_AGN[j]=float(len(ind3[0]))/float(len(ind_AGN_test))
        print(len(ind3[0]),len(ind_new_AGN[0]),len(ind_AGN_test),accuracy_AGN[j],completeness_AGN[j])

    test4=type_arr_test[ind_new_liner]
    ind4=np.where(test4==4)
    if len(ind_new_liner[0])==0:
        accuracy_liner[j]=0.
        completeness_liner[j]=0.
        print(len(ind4[0]),len(ind_new_liner[0]),accuracy_liner[j])
    else:
        accuracy_liner[j]=float(len(ind4[0]))/float(len(ind_new_liner[0]))
        completeness_liner[j]=float(len(ind4[0]))/float(len(ind_liner_test))
        print(len(ind4[0]),len(ind_new_liner[0]),len(ind_liner_test),accuracy_liner[j],completeness_liner[j])

accuracy_overall=(accuracy_AGN+accuracy_sf+accuracy_comp+accuracy_liner)/4.

alpha=0.5
plt.figure(0,figsize=(9.5,10))
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

plt.subplot(221)

plt.plot(mark_points,accuracy_sf,'b',alpha=alpha)
plt.plot(mark_points,accuracy_sf,'b+',label="SFGs",alpha=alpha)
plt.plot(mark_points,accuracy_comp,'g',alpha=alpha)
plt.plot(mark_points,accuracy_comp,'gv',marker='v',label="Composites",alpha=alpha)
plt.plot(mark_points,accuracy_AGN,'r',alpha=alpha)
plt.plot(mark_points,accuracy_AGN,'rd',label="AGNs",alpha=alpha)
plt.plot(mark_points,accuracy_liner,'y',alpha=alpha)
plt.plot(mark_points,accuracy_liner,'yx',label="LINERs",alpha=alpha)
plt.plot(mark_points,accuracy_overall,'m',label="Overall",alpha=alpha)
plt.axis([0,mark_points[-1], 0, 1])
plt.xlabel('Training Sample Size')
plt.ylabel('Classification Accuracy')
plt.title(title)
plt.legend(loc='lower right')



plt.subplot(222)
plt.plot(mark_points,completeness_sf,'b',linestyle='dashed',label="SFGs",alpha=alpha)
plt.plot(mark_points,completeness_comp,'g',linestyle='dashed',label="Composites",alpha=alpha)
plt.plot(mark_points,completeness_AGN,'r',linestyle='dashed',label="AGNs",alpha=alpha)
plt.plot(mark_points,completeness_liner,'y',linestyle='dashed',label="LINERs",alpha=alpha)
plt.axis([0,mark_points[-1], 0, 1])
plt.xlabel('Training Sample Size')
plt.ylabel('Classification Completeness')
plt.title(title)
plt.legend(loc='lower right')


plt.subplot(223)
plt.scatter(accuracy_sf,completeness_sf,c='b',marker='+',label="SFGs",alpha=alpha)
plt.scatter(accuracy_comp,completeness_comp,c='g',marker='v',label="Composites",alpha=alpha)
plt.scatter(accuracy_AGN,completeness_AGN,c='r',marker='d',label="AGNs",alpha=alpha)
plt.scatter(accuracy_liner,completeness_liner,c='y',marker='x',label="LINERs",alpha=alpha)
plt.axis([0.3,1, 0, 1])
plt.xlabel('Accuracy')
plt.ylabel('Completeness')
plt.legend(loc='lower right')

plt.subplot(224)
plt.scatter(accuracy_sf,accuracy_comp,c='g',marker='v',label="SF vs Composites",alpha=alpha)
plt.scatter(accuracy_sf,accuracy_AGN,c='r',marker='d',label="SF vs AGNs",alpha=alpha)
plt.scatter(accuracy_sf,accuracy_liner,c='y',marker='x',label="SF vs LINERs",alpha=alpha)
plt.axis([0.85,1, 0.3, 1])
plt.xlabel('SF Accuracy')
plt.ylabel('Other Accuracy')
plt.legend(loc='lower right')


plt.show()


print(accuracy_overall)

