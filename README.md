# bpl-next
![tests](https://github.com/anguswilliams91/bpl-next/actions/workflows/tests.yml/badge.svg)

new version of [bpl](https://github.com/anguswilliams91/bpl/tree/master), implemented in numpyro

## Statistical model

The statistical model behind `bpl` is a slight variation on the Dixon & Coles approach.
The likelihood is:

![equation](https://latex.codecogs.com/gif.latex?p(y_h%2C%20y_a)%20%3D%20\tau(y_h%2C%20y_a)\times%20\mathrm{Poisson}(y_h%20\%2C%20|%20\%2C%20a_h%20b_a%20\gamma_h)%20\times%20\mathrm{Poisson}(y_a%20\%2C%20|%20\%2C%20a_a%20b_h))

where y_h and y_a are the number of goals scored by the home team and the away team, respectively.
a_i is the *attacking aptitude* of team i and b_i is the *defending aptitude* of team j.
gamma_i represents the home advantage for team i, and tau is a correlation term that was introduced by Dixon and Coles to produce more realistic scorelines in low-scoring matches.
The model uses the following bivariate, hierarchical prior for a and b

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20%5Clog%20a_i%20%5C%5C%20%5Clog%20b_i%20%5Cend%7Bbmatrix%7D%20%5C%2C%20%5Cbig%20%7C%20%5C%2C%20X_i%5Csim%20%5Cmathcal%7BN%7D%20%5Cleft%28%20%5Cbegin%7Bbmatrix%7D%20X_i%20.%20%5Cbeta_a%20%5C%5C%20%5Cmu_b%20&plus;%20X_i%20.%20%5Cbeta_b%20%5Cend%7Bbmatrix%7D%2C%5Cquad%20%5Cbegin%7Bbmatrix%7D%20%5Csigma_a%5E2%2C%20%5Cquad%20%5Crho%20%5Csigma_a%20%5Csigma_b%20%5C%5C%20%5Crho%20%5Csigma_a%20%5Csigma_b%2C%20%5Cquad%20%5Csigma_b%5E2%20%5Cend%7Bbmatrix%7D%20%5Cright%29.)

X_i are a set of (optional) team-level covariates (these could be, for example, the attack and defence ratings of team i on Fifa).
beta are coefficient vectors, and mu_b is an offset for the defence parameter.
rho encodes the correlation between a and b, since teams that are strong at attacking also tend to be strong at defending as well.
The home advantage has a log-normal prior

![equation](https://latex.codecogs.com/gif.latex?\gamma_i%20\sim%20\mathrm{LogNormal}(\mu_\gamma%2C%20\sigma_\gamma)%2C)


Finally, the hyper-priors are

![equation](https://latex.codecogs.com/gif.latex?\begin{align}%20\mu_b%2C%20\beta_a%2C%20\beta_b%2C\mu_\gamma%20%26\sim%20\mathcal{N}(0%2C%201)%2C%20\nonumber%20\\%20\sigma_a%2C%20\sigma_b%20%2C%20\sigma_\gamma%20%26\sim%20\mathcal{N}^&plus;(0%2C%201)%2C%20\nonumber%20\\%20u%20%3D%20(\rho%20&plus;%201)%20/%202%20%26\sim%20\mathrm{Beta}(2%2C%204).%20\nonumber%20\end{align})
