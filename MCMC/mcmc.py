import emcee
import triangle
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt


# Reproductablity
np.random.seed(123)

# Choose the fiducial parameters.
m_true = -0.9594
b_true = 4.294

# Generate some data from a simple model y = m*x + b.
N = 50
xdata = np.sort(10*np.random.rand(N))
y_true = m_true*xdata+b_true
# Add some noise
ydata = y_true + np.random.randn(N)*0.5
yerr = np.ones_like(y_true)*0.1*ydata.max()


xl = np.array([0, 10])
plt.errorbar(xdata, ydata, yerr=yerr, fmt=".k", label='Observed')
plt.plot(xl, m_true*xl+b_true, "r", lw=3, alpha=0.6, label='True')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.savefig("line-data.png")
plt.show()


# Define the prior on our parameters m and b
def lnprior(theta):
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        return -np.log(0.5+5) - np.log(10.) 
    return -np.inf #log(0)

def lnlike(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

ndim = 2
nwalkers = 100
nburn = 1000
nsteps = 2000

starting_guesses = np.random.random((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[xdata, ydata, yerr])
sampler.run_mcmc(starting_guesses, nsteps)
print("done")

samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

fig = triangle.corner(samples, labels=["$m$", "$b$"],
                      truths=[m_true, b_true])
fig.savefig("line-triangle.png")
plt.show()




print samples[:, 1]
print samples[:, 0]

m_mean = np.mean(samples[:, 0])
b_mean = np.mean(samples[:, 1])
m_std = np.std(samples[:, 0])
b_std = np.std(samples[:, 1])

plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8, label='True')
plt.plot(xl, m_mean*xl+b_mean, color="b", lw=2, alpha=0.8, label="MCMC fit")
plt.errorbar(xdata, ydata, yerr=yerr, fmt=".k", label='Observed')
plt.ylim(-9, 9)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout()
plt.legend()
plt.savefig("line-mcmc.png")
plt.show()

# Compute the quantiles.
m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                               axis=0)))
#print("""MCMC result:
#    m = {0[0]} +{0[1]} -{0[2]} (truth: {1})
#    b = {2[0]} +{2[1]} -{2[2]} (truth: {3})
#""".format(m_mcmc, m_true, b_mcmc, b_true))

print("""MCMC result:
    m = {0} +/- {1} (truth {2})
    b = {3} +/- {4} (truth {5})
""".format(m_mean, m_std, m_true, b_mean, b_std, b_true))

import pymc

m = pymc.Uniform(name='m', lower=-5, upper=0.5)
b = pymc.Uniform(name='b', lower=0, upper=10)

@pymc.deterministic
def y_model(x=xdata, m=m, b=b):
    return b + m * x

y = pymc.Normal('y', mu=y_model, tau=1. / yerr ** 2, observed=True, value=ydata)
# package the full model in a dictionary
model1 = dict(m=m, b=b, y_model=y_model, y=y)

# run the basic MCMC: we'll do 100000 iterations to match emcee above
MDL = pymc.MCMC(model1)
MDL.sample(iter=100000, burn=50000)


# extract and plot results

m_pymc = np.mean(MDL.trace('m')[:])
b_pymc = np.mean(MDL.trace('b')[:])
m_std_pymc = np.std(MDL.trace('m')[:])
b_std_pymc = np.std(MDL.trace('b')[:])

print ("""PYMC result:
         m = {0} +/- {1} (truth {2})
         b = {3} +/- {4} (truth {5})
""".format(m_pymc, m_std_pymc, m_true, b_pymc, b_std_pymc, b_true))

y_min = MDL.stats()['y_model']['quantiles'][2.5]
y_max = MDL.stats()['y_model']['quantiles'][97.5]
y_fit = MDL.stats()['y_model']['mean']
plt.plot(xl,m_true*xl+b_true,'r', lw=2, alpha=0.8, label='True')
plt.errorbar(xdata, ydata, yerr=yerr, fmt=".k", label="Observed")
plt.plot(xdata, y_fit, color="b", lw=2, alpha=0.8, label="MCMC fit")
plt.fill_between(xdata, y_min, y_max, color='0.5', alpha=0.5)
plt.legend()
plt.savefig("line-mcmc-2.png")
plt.show()


