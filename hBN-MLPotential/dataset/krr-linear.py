import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

a      = np.loadtxt("./E-BN.txt")
X      = a[:,0:121]      # training data
Y      = a[:,-1]         # target

##### normalization ##############
Xstd   = np.std(X,axis=0)
Xstd[120]   = 1.0
Xa     = np.mean(X,axis=0) 
Xn     = (X-Xa)/Xstd
Xn[:,120]=1
##################################
### Make sure the alpha value by CV
parameters = {'alpha':[1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]}
KRR        = KernelRidge(alpha=0.000001,kernel='linear')
cv         = GridSearchCV(KRR, parameters, cv=5)
cv.fit(Xn,Y)
##### training ###################
clf    = cv.best_estimator_
krr    = clf.fit(Xn,Y)
Y_pred = krr.predict(Xn)
##### some basic score######
R2     = krr.score(Xn,Y)
r2     = r2_score(Y,Y_pred)
RMSE   = np.sqrt(mean_squared_error(Y,Y_pred))
MAE    = mean_absolute_error(Y,Y_pred)
print("Result for the Energy Prediction:")
print("RMSE is %f" % RMSE)
print("MAE  is %f" % MAE)
print("R2   is %f" % r2)
##### get the parameters####
omega  = krr.dual_coef_
alpha0 = np.asmatrix(omega)*Xn 
alpha  = np.asmatrix(omega)*Xn/Xstd

######## how to calculate the energy by the parameters ###########
Y_p2   = alpha*(a[:,0:121]-Xa).T+alpha[0,120]
#r2   = r2_score(Y,Y_p2.T)

'''
##### predict the force##### 
b      = np.loadtxt("../FP-F.txt")
Xb     = b[:,1:122]
Yb     = b[:,-1]
Yb_p   = alpha*Xb.T

r2     = r2_score(Yb,Yb_p.T)
RMSE   = np.sqrt(mean_squared_error(Yb,Yb_p.T))
MAE    = mean_absolute_error(Yb,Yb_p.T)
print("Result for the Force Prediction:")
print("RMSE is %f" % RMSE)
print("MAE  is %f" % MAE)
print("R2   is %f" % r2)
'''
######## Output the prediction of energy #################
f = open("Energy.log","w")
for i in range(len(Y)):
    f.write("%.20lg  %.20lg  %.20lg\n"%(Y[i],Y_pred[i],Y_p2[:,i]))
f.close


######## Output the potential files ######################
f = open("GeSe.potential","w")
f.write("generation 2\n")
f.write("n_elements 2\n\n")
f.write("element B N\n\n")
f.write("N_flag 0\n\n")
f.write("interaction B\n")
f.write("Rc 5.5\n")
f.write("Rs 5.5\n")
f.write("Rm 5.5\n")
f.write("eta 8 1.3291\n")
#f.write("epsilon 1 2 4 8 16 32\n")
#f.write("Mu 2.50 2.90 3.25 4.11 4.60 5.00 5.40 6.00\n")
f.write("C %.20lg\n" % (alpha0[0,-1]))
f.write("alpha ")
n,m = alpha0.shape
for i in range(m):
    f.write("%.20lg " % (alpha0[0,i]))
f.write("\n")
f.write("Std_dev ")
for i in range(m):
    f.write("%.20lg " % (1/Xstd[i]))
f.write("\n")
f.write("mean ")
for i in range(m):
    f.write("%.20lg " % (Xa[i]))
f.write("\n")
f.write("endVar\n\n")

f.write("interaction N\n")
f.write("Rc 5.5\n")
f.write("Rs 5.5\n")
f.write("Rm 5.5\n")
f.write("eta 8 1.3291\n")
#f.write("epsilon 1 2 4 8 16 32\n")
#f.write("Mu 2.50 2.90 3.25 4.11 4.60 5.00 5.40 6.00\n")
f.write("C %.20lg\n" % (alpha0[0,-1]))
f.write("alpha ")
n,m = alpha0.shape
for i in range(m):
    f.write("%.20lg " % (alpha0[0,i]))
f.write("\n")
f.write("Std_dev ")
for i in range(m):
    f.write("%.20lg " % (1/Xstd[i]))
f.write("\n")
f.write("mean ")
for i in range(m):
    f.write("%.20lg " % (Xa[i]))
f.write("\n")
f.write("endVar\n\n")
f.close()

























