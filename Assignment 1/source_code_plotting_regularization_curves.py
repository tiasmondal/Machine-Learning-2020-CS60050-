import matplotlib.pyplot as plt
import numpy as np
train=np.array([0.0995745390593665,0.09945575873293316,0.09933725981541953,0.09921903470815716])
test=np.array([0.09488619793431836,0.09430315047432938,0.09371869235219425,0.0931328377692365])
lambda1=np.array([0.25,0.5,0.75,1])
test=test.reshape((4,1))
train=train.reshape((4,1))
lambda1=lambda1.reshape((4,1))

x=plt.plot(lambda1,train,Label="Training error")
y=plt.plot(lambda1,test,Label="Test error")
plt.legend()
plt.xlabel("value of lambda")
plt.ylabel("error")
plt.title("Order 1")

plt.show()