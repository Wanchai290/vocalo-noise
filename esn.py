import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import ESN

import numpy as np
import matplotlib.pyplot as plt

rpy.set_seed(42)  # make everything reproducible!

##datas
X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
X_train = X[:50]
Y_train = X[1:51]

##model
reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = ESN(reservoir, ridge)

##train
esn_model = esn_model.fit(X_train, Y_train, warmup=10)
print(reservoir.initialized, ridge.initialized)


##run model
Y_pred = np.empty((100, 1))
x = Y_train[-1]

for i in range(100):
    x = esn_model(x)
    Y_pred[i] = x



plt.figure(figsize=(10, 3))
plt.title("100 timesteps of a sine wave.")
plt.xlabel("$t$")
plt.plot(Y_pred, label="Generated sin(t)")
plt.legend()
plt.show()

