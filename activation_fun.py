
import numpy as np
import matplotlib.pyplot as plt


x=list(range(-10,11,1))


X=-1*np.array(x)
tmp=np.exp(X)
y1=1/(1+tmp)

plt.plot(x,y1,label='sigmoid')
plt.legend()
plt.show()


y2=(1-np.square(tmp))/(1+np.square(tmp))

plt.plot(x,y2,label='tanh')
plt.legend()
plt.show()


y3=(np.array(x) > 0)

plt.plot(x,y3,label='unit step uniploar')
plt.legend()
plt.show()


y4=np.where(np.array(x)>0,1,-1)

plt.plot(x,y4,label='unit step biploar')
plt.legend()
plt.show()


