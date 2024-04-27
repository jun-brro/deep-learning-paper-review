# Week 2. StyleGAN

작성날짜: 2024년 3월 19일

작성자: Junhyeong Park

# Base: PGGAN

<img width="697" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/c0838342-f24f-4c0d-bec3-43061057e24b">


 **점진적으로 layer를 추가하면서 학습 → 안정적으로 고해상도 이미지를 만들 수 있다.**

- **Progressive Training:** Starts with low-resolution images, gradually increases resolution by adding layers, improving stability and image quality.
- **Improved Image Quality:** Generates high-resolution, high-quality images.
- **Training Stability:** Reduces instability in training high-resolution images.
- **Adaptive Learning Rate:** Adjusts learning rates for layers; new layers have higher rates.
- **Normalization Techniques:** Uses Pixelwise Feature Vector Normalization in the generator for stable training.
- **Enhanced Detail Capture:** Effectively captures fine details for more realistic images.
- **Flexibility Across Domains:** Proven effective in various domains, not just faces.
- **Influences Research:** Influenced subsequent GAN research, promoting progressive training and high-resolution generation.

# Style-based Generator

<img width="822" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/808eb66e-c516-4201-aacd-99b0e362702d">

- GAN에서는 생성자가 input 레이어를 통해 코드를 받지만, StyleGAN은 input 레이어를 생략하고 학습된 상수에서 intermediate latent space W로의 비선형 매핑을 사용하여 시작한다.
- **Intermediate Latent Space W:** input latent space Z의 latent code z가 nonlinear mapping network f를 통해 intermediate latent space W의 w로 변환된다. (8층 MLP 사용하여 구현)
- **Style Transfer:** The converted vector w is specialized into a spatially invariant style y through learned affine transformations. This style y then controls the Adaptive Instance Normalization (AdaIN) at each convolution layer of the generator.
    
<img width="655" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/e57716fe-908c-45b7-b0a1-2b7138bf0d57">
    

<img width="695" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/bf6ee61d-e861-4476-a3be-cf9a630d9afd">

The AdaIN operation normalizes each feature map separately and then scales and biases it using the corresponding scalar components of style y. This means that for each feature map, AdaIN adjusts its scale and shifts based on the style information encoded in y. This process allows the generator to apply complex style patterns to the generated images, influencing everything from color schemes to textures in a coherent and controlled manner. The use of spatially invariant style means that the style is uniformly applied across the entire image, rather than being dependent on the spatial location within the image, which is crucial for ensuring the generated image maintains a consistent style throughout.

- **Noise Input:** To enhance the generator's ability to create stochastic details, we've implemented explicit noise inputs. These inputs are single-channel images filled with uncorrelated Gaussian noise. Each layer of the synthesis network receives its own unique noise image. The noise is then scaled for each feature map based on learned scaling factors and added to the output of its respective convolution layer.

<img width="293" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/f23ed6e1-a98e-447c-b603-2e707f8b1ed7">

### Quality of generated images

- Improvement of FID (Frechet Inception Distances): The improvement in image quality is substantiated by comparing FID across different generator architectures for the CELEBA-HQ and the new FFHQ datasets.
- Further, architectural advancement and enhancement of image resolution is achieved.

### Prior Art

- Focus on Discriminator Improvements
    - multiple discriminators, multi-resolution discrimination, self-attention
- Research on the Generator
    - Concentrated on fine-tuning the distribution in the input latent space or shaping the latent space, with conditional generators exploring new methods of feeding class identifiers through several layers of the generator

# Properties of the style-based generator

### Style Mixing

<img width="795" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/bb33d2d1-e774-4e62-9f6d-f07bd9e689a5">

**Mixing Regularization:** Using ‘mixing’ during learning process as ‘regularization’

**Style Mixing:** Using same mixing during test → able to mix styles of two different images

### Stochastic variation

<img width="431" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/20f20554-eee4-495f-a5c9-0ee3018e759f">

<img width="416" alt="Untitled 7" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2a99646c-5964-4536-aa8c-f6d6ece78598">

### Separation of global effects from stochasticity

# Disentanglement Studies

### Perceptual Path Length

<img width="322" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/ad13296e-8e76-466b-8ee5-0c3b8a797224">

<img width="404" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/298a1098-4644-43e0-b161-b5191c2e8fef">

<img width="341" alt="Untitled 10" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a8571e11-5ea9-4f18-816d-f8cc21fa54e5">

### Linear Separability

# Conclusion
