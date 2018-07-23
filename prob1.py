import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

n = 50
omega = np.random.rand()
noise = 0.8 * np.random.randn(n)
x = np.random.randn(n,2)
y = 2*(omega*x[:,0] + x[:,1] + noise > 0) - 1


w1 = w2 = np.array([0.0, 1.0])
epoch = 100
J1_list = []
J2_list = []
w1_list = []
hessian_c = np.zeros((2, 2))

for i in range(epoch):
    J1, grad, J2, grad2 = 0, 0, 0, 0
    d1, d2 = 0, 0
    _lambda = 0.01

    for j in range(n-1):
        a = np.exp(-1*y[j]*np.dot(w1,x[j]))
        a2 = np.exp(-1*y[j]*np.dot(w2,x[j]))
        grad += a*(-1*y[j]*x[j])/(1+a)
        grad2 += a2*(-1*y[j]*x[j])/(1+a2)
        hessian_c += a2*np.dot(x[j][:,None],x[j][:,None].T)/((1+a)**2)
        J1 += np.log(1+np.exp(-1*y[j]*np.dot(w1,x[j])))
        J2 += np.log(1+np.exp(-1*y[j]*np.dot(w2,x[j])))
    J1 = J1 + _lambda*np.dot(w1,w1)
    J2 = J2 + _lambda*np.dot(w2,w2)
    d1 = -1*grad/n + 2*_lambda*w1
    d2 = -1*np.dot(np.linalg.inv(hessian_c/n + 2*_lambda*np.identity(2))
            , grad2/n + 2*_lambda*w2)
    w1 = w1 + 0.1*d1
    w2 = w2 + 0.1*d2
    J1_list.append(J1)
    J2_list.append(J2)


# matplotlib.pyplot

fig = plt.figure()
axR = fig.add_subplot(211)
axL = fig.add_subplot(212)

axR.plot(np.arange(epoch),np.array(J1_list))
axR.plot(np.arange(epoch),np.array(J2_list))
axR.set_xlabel("epoch")

pl = np.empty((0, 2), int)
pl2 = np.empty((0, 2), int)
for i in np.arange(-2.5, 2.5, 0.01):
    for j in np.arange(-2.5, 2.5, 0.01):
        coo = np.array([[i, j]])
        if abs(np.dot(w1,coo[0])) < 0.01: pl = np.append(pl, coo, axis=0)
        if abs(np.dot(w2,coo[0])) < 0.01: pl2 = np.append(pl2, coo, axis=0)
axL.plot(pl[:,0], pl[:,1])
axL.plot(pl2[:,0], pl2[:,1])
axL.scatter(x[:,0],x[:,1],c=y,s=50)


plt.show()



