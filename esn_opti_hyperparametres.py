import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass
from reservoirpy import ESN

import numpy as np
import matplotlib.pyplot as plt

##params
UNITS = 100               # - number of neurons
LEAK_RATE = 0.3           # - leaking rate
SPECTRAL_RADIUS = 1.25    # - spectral radius of W
INPUT_SCALING = 1.0       # - input scaling
RC_CONNECTIVITY = 0.1     # - density of reservoir internal matrix
INPUT_CONNECTIVITY = 0.2  # and of reservoir input matrix
REGULARIZATION = 1e-8     # - regularization coefficient for ridge regression
SEED = 42

rpy.set_seed(SEED)  # make everything reproducible!

##datas
X = mackey_glass(2000)
X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

plt.figure()
plt.xlabel("$t$")
plt.title("Mackey-Glass timeseries")
plt.plot(X[:500])
plt.show()

##spectral radius 
states = []
spectral_radii = [0.1, 1.25, 10.0]
for spectral_radius in spectral_radii:
    reservoir = Reservoir(
        units=UNITS, 
        sr=spectral_radius, 
        input_scaling=INPUT_SCALING, 
        lr=LEAK_RATE, 
        rc_connectivity=RC_CONNECTIVITY,
        input_connectivity=INPUT_CONNECTIVITY,
        seed=SEED,
    )

    s = reservoir.run(X[:500])
    states.append(s)

UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))
for i, s in enumerate(states):
    plt.subplot(len(spectral_radii), 1, i+1)
    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)
    plt.ylabel(f"$sr={spectral_radii[i]}$")
plt.xlabel(f"Activations ({UNITS_SHOWN} neurons)")
plt.show()


##Input scaling
states = []
input_scalings = [0.1, 1.0, 10.]
for input_scaling in input_scalings:
    reservoir = Reservoir(
        units=UNITS, 
        sr=SPECTRAL_RADIUS, 
        input_scaling=input_scaling, 
        lr=LEAK_RATE,
        rc_connectivity=RC_CONNECTIVITY, 
        input_connectivity=INPUT_CONNECTIVITY, 
        seed=SEED,
    )

    s = reservoir.run(X[:500])
    states.append(s)

UNITS_SHOWN = 20

plt.figure(figsize=(15, 8))
for i, s in enumerate(states):
    plt.subplot(len(input_scalings), 1, i+1)
    plt.plot(s[:, :UNITS_SHOWN], alpha=0.6)
    plt.ylabel(f"$iss={input_scalings[i]}$")
plt.xlabel(f"Activations ({UNITS_SHOWN} neurons)")
plt.show()


# X_train = X[:50]
# Y_train = X[1:51]

# ##model
# reservoir = Reservoir(100, lr=0.5, sr=0.9)
# ridge = Ridge(ridge=1e-7)

# esn_model = ESN(reservoir, ridge)

# ##train
# esn_model = esn_model.fit(X_train, Y_train, warmup=10)
# print(reservoir.initialized, ridge.initialized)


# ##run model
# Y_pred = np.empty((100, 1))
# x = Y_train[-1]

# for i in range(100):
#     x = esn_model(x)
#     Y_pred[i] = x



# plt.figure(figsize=(10, 3))
# plt.title("100 timesteps of a sine wave.")
# plt.xlabel("$t$")
# plt.plot(Y_pred, label="Generated sin(t)")
# plt.legend()
# plt.show()

