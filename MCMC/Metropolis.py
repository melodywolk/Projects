import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



def generate(initial_state, nb_iterations, pdf):
    # Initialization
    list_state = []
    state = initial_state
    # Check the size of the new state to be generated
    if (type(state)==float or type(state)==int):
       dim = 1
    else:
       dim = len(state)
    for i in range(0, nb_iterations):
        # Draw a candidate state of size dim from a multivariate normal distribution (our proposal)
        candidate = state + stats.multivariate_normal.rvs(size=dim)
        uniform = stats.uniform(0,1).rvs()
        ratio = pdf(candidate)/pdf(state)
        if (uniform < ratio):
           state = candidate
        else:
           state = state
        list_state.append(state)

    return list_state


pdf = stats.norm().pdf
initial_value = 0
n_iterations = 10000

vals = generate(initial_value, n_iterations, pdf)

plt.hist(vals, bins=20, label='MCMC')
plt.hist(np.random.normal(size=n_iterations), bins=20, alpha=0.5, label='PDF')
plt.legend()
plt.savefig("Metropolis.png")
plt.show()
        
       

        




