# Week 3. VAE

작성날짜: 2024년 3월 26일

작성자: Junhyeong Park

### 참고 자료
[VAE : Auto-Encoding Variational Bayes - 논문 리뷰](https://velog.io/@lee9843/VAE-Auto-Encoding-Variational-Bayes-논문-리뷰)

# Introdction

How can we perform efficient approximate inference and learning with directed probabilistic models whose continuous latent variables and/or parameters have intractable posterior distributions? 

Answer : **Variational Bayesian** (Includes optimization of approximations to intractable posterior probabilities)

### Variational Bayesian (Appendix F)

Marginal Likelihood: KL divergence + lower bound

<img width="467" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/fc4cbed7-6b60-40ea-a9c7-c5c9fdb4ba59">


<img width="489" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4edcf087-3b7e-4073-9063-b705bdbc8f99">


Variational lower bound to the marginal likelihood

<img width="490" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/102829cb-e02d-49ce-bf67-f8da7b6058cd">

<img width="493" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a07c0c2f-92dc-4628-abee-88b2d0706ce0">

…

Monte Carlo estimate of the variational lower bound

<img width="299" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/c9c61a64-253d-4607-acda-1466ca006569">

[Variational Bayesian methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)

### Stochastic Gradient Variational Bayes (SGVB)

- *SGVB estimator can be used for efficient approximate posterior inference in almost any model with continuous latent variables and/or parameters, and is straightforward to optimize using standard stochastic gradient ascent techniques.*
- A scalable estimator for variational inference that utilizes stochastic gradients, enabling optimization over large datasets.
- Facilitates efficient backpropagation through recognition models by approximating gradients.

### Auto-Encoding Variational Bayes (AEVB)

- *In the AEVB algorithm we make inference and learning especially efficient by using the SGVB estimator to optimize a recognition model that allows us to perform very efficient approximate posterior inference using simple ancestral sampling, which in turn allows us to efficiently learn the model parameters, without the need of expensive iterative inference schemes (such as MCMC) per datapoint.*
- An algorithm that pairs with neural network architectures to create variational auto-encoders.
- It specifically tailors the variational inference process for datasets with continuous latent variables.

# Method

<img width="530" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/5d0cf6c3-2880-428d-aec0-4512a9bc64ae">

### Problem Scenario

Considering the dataset below

<img width="314" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4a64183e-7f05-4497-83bc-970973bfb694">

1. The latent variable  z^i is generated from the prior distribution p_theta(z).
2. The dataset x^i is generated from the conditional distribution p_theta(x|z).

![Untitled 7](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/e9a6c5fa-d805-425e-872c-f5675c77f4eb)

*Very importantly, we do not make the common simplifying assumptions about the marginal or posterior probabilities → Taking care of*

1. **Intractability** (cannot compute marginal likelihood)
2. **A large dataset** (If too large dataset, the sampling should be conducted for each data points → Too much cost for batch optimization)

The research propose a solution to, three problems considering the scenario above:

1. *Efficient approximate ML or MAP estimation for the parameters θ. The parameters can be of interest themselves, e.g. if we are analyzing some natural process. They also allow us to mimic the hidden random process and generate artificial data that resembles the real data.*
2. *Efficient approximate posterior inference of the latent variable z given an observed value x for a choice of parameters θ. This is useful for coding or data representation tasks.*
3. *Efficient approximate marginal inference of the variable x. This allows us to perform all kinds of inference tasks where a prior over x is required. Common applications in computer vision include image denoising, inpainting and super-resolution.*

To solve all those problems, the study introduces **a recognition model q_theta(z|x) (*an approximation to the intractable true posterior p_theta(z|x)***

**METHOD SUMMARY**

- The recognition model parameters *ϕ* are learned together with the generative model parameters *θ*.
- Given a data point *x*, **a stochastic encoder** produces a distribution (e.g., Gaussian) of possible values for the code *z* that could generate *x*.
- *p_θ*(*x*∣*z*) is **a stochastic decoder** that produces a distribution of possible values of *x* given *z*.

<img width="576" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/019bf071-e7d3-4209-950e-6d160210b558">

### The variational bound

Marginal Likelihood log(p_theta(x^i))

<img width="448" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/49db5dae-ee1f-45c4-8c91-85161c6b6b23">

RHS

- 1st: KL divergence of the approximate from the true posterior (non-negative)
- 2nd: L(theta, pi;x^i), (variational) lower bound on the marginal likelihood of datapoint i
    
<img width="451" alt="Untitled 10" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/fb3ec5c3-f855-4c93-816e-3bf8b3a9a64d">
    
<img width="439" alt="Untitled 11" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/60b1cc4f-2e9b-4ff9-b0e0-85d35c9252cb">
    

Objective: to differentiate and optimize the lower bound.

<img width="587" alt="Untitled 12" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/c243f655-27a7-4e8a-8e68-24756dd07c9e">

This corresponds to calculating the probability of *x* in Bayes' theorem, which is the marginal likelihood or evidence, and hence it is called the Evidence Lower BOund, or **ELBO**. Loss function is made by this ELBO.

<img width="591" alt="Untitled 13" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6470897e-6044-4b08-b888-eca7da3c6060">

### The SGVB estimator and AEVB algorithm

<img width="535" alt="Untitled 14" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/fe40492d-7a49-4cac-ab79-2b049f529a8d">

### Reparameterization Trick

<img width="594" alt="Untitled 15" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a99833ee-ffb3-4001-b151-8025747cff67">

Two assumptions needed to compute regularization:

- The distribution of z that emerges from passing through the encoder,  q_phi(z|x), follows a multivariate normal distribution with a diagonal covariance matrix.
- The assumed distribution of z, the prior p(z), is that it follows a standard normal distribution with a mean of 0 and a standard deviation of 1.

→ KLD makes those to assumptions same, and also handle optimization.

<img width="594" alt="Untitled 16" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/7f263752-d75d-46fd-be1b-6162b0bc9364">

mu_1 = 0, sigma_1 = 1 (average 0, standard deviation 1)

<img width="605" alt="Untitled 17" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6371414c-b85b-4d62-b962-bf3d8d8501d7">

Thus it is differentiable, therefore able to calculate regularization.

# Variational Auto-Encoder (VAE)

Let the variational approximate posterior be a multivariate Gaussian with a diagonal covariance structure

<img width="383" alt="Untitled 18" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/8f0c7e31-1e86-4efb-b6fe-4ad6a1f67a55">

<img width="493" alt="Untitled 19" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/1a26c6a5-2ac3-417b-9e18-f8a084e36157">

log(p_theta(x^i | z^(i, l))) is a Bernoulli or Gaussian MLP, depending on the type of data.

### Appendix C: MLP’s as probabilistic encoders and decoders

**Bernoulli MLP**

<img width="493" alt="Untitled 20" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/45110c25-7f51-4e49-8a9b-ff277db35107">

<img width="592" alt="Untitled 21" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2e849014-0a98-4e90-b443-8cfd5170e81f">

---

**Gaussian MLP**

<img width="507" alt="Untitled 22" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/bede1ba3-a3be-4d91-b61a-d7695a60ae50">

<img width="592" alt="Untitled 23" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/ffc15c2a-2085-4789-9a0c-9186e0755a60">

<img width="592" alt="Untitled 24" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2c1856a6-7090-473e-a357-11d2c57bc6cb">

<img width="609" alt="Untitled 25" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/eab9ee56-f060-4e74-9a85-3b893fc0d6e5">

<img width="583" alt="Untitled 26" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/cbd573ad-2aef-4e65-953e-9ab5f30a00eb">

# Experiments

MNIST & Frey Face datasets

<img width="531" alt="Untitled 27" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/95d86097-c3a2-4450-9745-eb7691b54a28">

### Likelihood lower bound

### Marginal likelihood

<img width="530" alt="Untitled 28" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2ee19c13-08cb-484e-925d-2dfc15f95450">

### Visualization of high-dimensional data

# Conclusion & Future Work

### Conclusion

- The SGVB estimator and AEVB algorithm significantly improve variational inference for continuous latent variables.
- Demonstrated theoretical advantages and experimental results.

### Future work

- **Hierarchical Generative Architectures**: Investigating the use of SGVB and AEVB in learning hierarchical generative models, particularly with deep neural networks such as convolutional networks for encoders and decoders.
- **Time-Series Models**: Applying these methods to dynamic Bayesian networks for modeling time-series data.
- **Global Parameters**: Extending the application of SGVB to optimize global parameters within models.
- **Supervised Models with Latent Variables**: Exploring supervised models that incorporate latent variables, aiming to learn complex noise distributions, which can enhance model robustness and predictive performance.
