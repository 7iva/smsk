import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 40
omega = np.random.rand()
noise = 0.8 * np.random.randn(n)
x = np.random.randn(n,2)
y = 2*(omega*x[:,0] + x[:,1] + noise > 0) - 1

"""
x=np.array([[-1,-0.7],[-1,-1.2],[0,0.1],[0,-0.1],[1,1.1],[1,0.9]])
y=np.array([1,-1,1,-1,1,-1])
n=6

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x[:,0],x[:,1],y, "o")
plt.show()
"""


#w = np.random.rand(2)
w = np.array([1.0,1.0])
epoch = 500
J_list = []
w_list = []

for i in range(epoch):
    jj, J, grad = 0, 0, 0
    h = 0

#    print(str(i) + 'th epoch')
    for j in range(n-1):
        a = np.exp(-1*y[j]*np.dot(w,x[j]))
        jj += a*(-1*y[j]*x[j])/(1+a)
        J += np.log(1+np.exp(-1*y[j]*np.dot(w,x[j])))
    J = J + np.dot(w,w)
    grad = jj/n + 2*0.01*w
#    print('grad=' + str(J))
    print('grad=')
    print(grad)
    w = w - 0.05*grad
    J_list.append(J)

print(np.dot(w,np.array([0.5, -0.5])))
print(np.dot(w,np.array([0, 1.0])))
print(np.dot(w,np.array([-1.0, 1.0])))

plt.plot(np.arange(epoch),np.array(J_list)) 
plt.xlabel("epoch")

plt.show()


