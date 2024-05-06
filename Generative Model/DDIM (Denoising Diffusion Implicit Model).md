# Week 5. DDIM

# Introduction - GAN vs DDPM

- GAN has faster ability than DDPM, in terms of image generation
    - *Deep generative models have demonstrated the ability to produce high quality samples in many domains (Karras et al., 2020; van den Oord et al., 2016a). In terms of image generation, generative adversarial networks (GANs, Goodfellow et al. (2014)) currently exhibits higher sample quality than likelihood-based methods such as variational autoencoders (Kingma & Welling, 2013), autoregressive models (van den Oord et al., 2016b) and normalizing flows (Rezende & Mohamed, 2015; Dinh et al., 2016). However, GANs require very specific choices in optimization and architectures in order to stabilize training (Arjovsky et al., 2017; Gulrajani et al., 2017; Karras et al., 2018; Brock et al., 2018), and could fail to cover modes of the data distribution (Zhao et al., 2018).*
- **DDIMs are implicit probabilistic models (Mohamed & Lakshminarayanan, 2016) and are closely related to DDPMs, in the sense that they are trained with the same objective function.**
- **Generalizing the Process:** A DDPM's forward diffusion process can be generalized from a Markovian process to a non-Markovian process, allowing for a compatible reverse Markov chain to be designed.
- **Objective Function:** The same objective function used to train a Markovian DDPM can be applied, allowing for the same neural network to be employed, regardless of the diffusion process type.
- **Flexibility:** This flexibility enables the use of various neural networks in different generative models.
- **Short Markov Chain:** The non-Markovian process facilitates a shorter generative Markov chain with fewer steps, greatly enhancing sampling efficiency with minimal quality loss.

<img width="685" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4345b553-0fab-42de-b3c7-86950c38e4a0">

---

# Background

[Week 4. DDPM](https://www.notion.so/Week-4-DDPM-e181db88a14f4a738e565a95950c3549?pvs=21)

---

# Variational Interference For Non-Markovian Forward Processes

## Non-Markovian Forward Processes

<img width="710" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6f67a32a-7021-4fec-863d-a378b67f5464">

- the forward process here is no longer Markovian, since each xt could depend on both xt−1 and x0.

## Generative Process And Unified Variance Interference Objective

<img width="727" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/8e98e8d7-e672-4fa2-a78d-169c08c5e43a">

## Theorem 1

<img width="709" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/8fd5d053-d6a4-4de7-abba-76f82ae2fc0b">

---

# Sampling From Generalized Generative Processes

<img width="535" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/e4fee028-e7fb-4680-a83c-ad62abfaa80b">

## Denoising Diffusion Implicit Models

$$
\begin{aligned}
x_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\bigg( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta^{(t)} (x_t)}{\sqrt{\alpha_t}} \bigg)}_{\textrm{predicted } x_0}
+ \underbrace{\sqrt{1-\alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta^{(t)} (x_t)}_{\textrm{direction pointing to } x_t}
+ \underbrace{\sigma_t \epsilon_t}_{\textrm{random noise}}
\end{aligned}
$$

- **Gaussian Noise:** t ~ N(0, I) represents standard Gaussian noise independent of x_t, with alpha_0 := 1.
- Different choices of σ values lead to different generative processes using the same model θ, removing the need for retraining.
- **Markovian Process:** When
    
    $$
    \begin{equation}
    \sigma_t = \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}} \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}}
    \end{equation}
    $$
    
    the forward process becomes Markovian, and the generative process is a DDPM.
    
- For σ_t = 0 for all t, the forward process becomes deterministic given x_(t-1) and x_0 (except for t = 1). In this generative process, the coefficient before the random noise t becomes zero.
- This results in an **implicit probabilistic model (DDIM)** where samples are generated from latent variables through a fixed procedure, trained with the DDPM objective, even though the forward process is no longer a diffusion.

## Accelerated Generation Processes

1. The generative process approximates the reverse of the inference process. When the inference process has T steps, the generative process is also constrained to sample T steps.
2. The generative model's denoising objective, L1, depends only on the fixed marginals qσ(xt|x0). This allows for forward processes with fewer steps, leading to shorter generative processes, thereby increasing efficiency.
3. The forward process is defined not over all latent variables x1:T, but over a subset {xτ1, ..., xτS}, which is a sequence of length S. This sequence maintains the marginals.
4. The generative process samples latent variables according to this shorter sequence, significantly increasing efficiency by reducing the number of sampling steps.
5. The L1 objective can justify training the model with its initial setup, allowing for new, faster generative processes without retraining.

## Relevance to Neural ODEs

$$
\begin{aligned}
x_{t-\Delta t} &= \sqrt{\alpha_{t-\Delta t}} \left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta^{(t)} (x_t)}{\sqrt{\alpha_t}}  \right) + \sqrt{1-\alpha_{t-\Delta t}} \cdot \epsilon_\theta^{(t)} (x_t) \\
&= \frac{\sqrt{\alpha_{t-\Delta t}}}{\sqrt{\alpha_t}} x_t + \sqrt{\alpha_{t-\Delta t}} \left( \sqrt{\frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}} - \sqrt{\frac{1-\alpha_t}{\alpha_t}} \right) \cdot \epsilon_\theta^{(t)} (x_t) \\
\frac{x_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} &= \frac{x_t}{\sqrt{\alpha_t}} + \left( \sqrt{\frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}} - \sqrt{\frac{1-\alpha_t}{\alpha_t}} \right) \epsilon_\theta^{(t)} (x_t)
\end{aligned}
$$

<img width="520" alt="image" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6d610ba6-0364-493f-aaa3-760539dabfa4">


The initial condition is x(T) ~ N(0, sigma(T)). From this, we can infer that by performing enough discretization steps, it is possible to reverse the ODE, which also makes it feasible to reverse the generation process, that is, to encode x_0 → x(T).

---

# Experiments

<img width="712" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/dae59b64-27a0-4d13-9b85-7889419434ae">

## Sample Quality and Efficiency

- The quality of generated samples was evaluated using the Frechet Inception Distance (FID) metric. It was found that as the number of sampling steps (dim(τ)) increases, sample quality improves, showing a trade-off between quality and computational cost.
- DDIM (η = 0) produced the best sample quality with small dim(τ), while DDPM typically had worse quality compared to its less stochastic counterparts, except for a specific case where dim(τ) = 1000.
- DDPM struggled more with smaller trajectories, while DDIM showed consistently better quality. In Figure 3, CIFAR10 and CelebA images generated by DDPM deteriorated rapidly with 10 steps, while DDIM remained consistent.

## Sample Consistency in DDIMs

<img width="733" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/d8ed11a2-1179-41db-bab1-f097920cca50">

## Interpolation in Deterministic Generative Processes

<img width="715" alt="Untitled 7" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/255fdb30-df1d-4729-ac03-f5a24a8a8982">

## Reconstruction from Latent Space

*As DDIM is the Euler integration for a particular ODE, it would be interesting to see whether it can encode from x0 to xT (reverse of Eq. (14)) and reconstruct x0 from the resulting xT (forward of Eq. (14))7.*
