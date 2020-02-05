---
title: "Learning from Simulated and Unsupervised Images through Adversarial Training"
published: true
use_math: true
categories: 
  - review
---

Learning from Simulated and Unsupervised Images through Adversarial Training (CVPR 2017)

Realistic한 합성 데이터를 만들어내는 게 목표
S+U (Simulated + Unsupervised) learning 기법을 사용한 SimGAN 이라는 GAN 소개

## SimGAN

**Simulator, Refiner, Discriminator**로 구성

**Simulator**: 합성 데이터 생성
 
**Refiner**: 생성된 이미지를 refine 해줌
  + Generator처럼 Adversarial loss 사용
  + 합성 data의 annotation을 유지하기 위해 추가적으로 self-regularization loss 사용
  + Self-regularization loss: Refine된 이미지가 합성 이미지와 너무 달라지는 것을 방지
  + 전체적인 구조를 유지하기 위해 Fully-Connected가 아닌 Pixel level에서 작동하는 Fully Convolutional 사용
    
**Discriminator**
  + Receptive field를 전체 이미지가 아닌 local region으로 제한
  + 이미지 당 여러개의 local adversarial loss 사용

## Learning SimGAN
SimGAN의 목표는 Unlabeled 이미지 $$\mathbf{y}_i \in \mathcal{Y}$$ 를 이용해 refiner $$R_{\theta}(\mathbf{x})$$ 를 학습시키는 것

$$\mathbf{x}$$ 는 Simulator에서 내놓은 합성 이미지

### Refiner

학습 과정에서 Refiner는 기존의 GAN에서 Generator와 비슷한 역할을 한다

Refiner의 loss는 realism loss($ \ell_{real}$)와 self-regularization loss($\ell_{reg}$)의 결합으로 이루어짐

$$
\mathcal{L}_R(\theta) = \sum_i \ell_{real}(\theta;\mathbf{x}_i,\mathcal{Y}) + \lambda\ell_{reg}(\theta;\mathbf{x}_i)
$$

$\ell_{real}$ 은 이미지가 realistic 하도록 만들어 주는 역할을 하고, $\ell_{reg}$는 이미지가 annotation 정보를 보존하도록 만들어 줌

**Realism Loss**

$\ell_{real}$ 은 기존의 GAN에서 Generator의 loss와 동일

$$
\ell_{real}(\theta;\mathbf{x}_i,\mathcal{Y}) = - \log(1 - D_\phi(R_\theta(\mathbf{x}_i)))
$$

**Self-regularization loss**

논문에서 $\ell_{reg}$ 는 refine 된 이미지와 기존의 합성 이미지의 per-pixel difference 를 최소화하는 것을 목표로 했음

$$\ell_{reg} = \|\psi(\tilde{\mathbf{x}}) - \psi(\mathbf{x})\|_1$$

$\psi$ 는 image space에서 feature space로 mapping 해주는 함수. Image derivatives나, 색상 평균, 혹은 CNN을 통한 변환 등 원하는 함수를 사용 가능하나, 논문에서는 단순한 identity map ($\psi(\mathbf{x}) = \mathbf{x}$) 사용.

결과적으로 Loss function은 다음과 같다

$$
\mathcal{L}_R(\theta) = -\sum_i \log(1 - D_\phi(R_\theta(\mathbf{x}_i))) + \lambda\|\psi(R_\theta(\mathbf{x}_i)) - \psi(\mathbf{x}_i)\|_1
$$

논문에서는 이미지를 전체적으로 수정하는 것이 아닌 픽셀 단위로 수정하기 위해 **$R_\theta$** 를 **striding이나 pooling이 없는 FCN**으로 구현했다고 함

### Discriminator

Discriminator의 목표는 refine 된 합성 이미지가 실제 이미지인지 아닌지 잘 구분하는 것

Loss는 기존 GAN의 Discriminator와 동일

$$
\mathcal{L}_D(\phi) = -\sum_i \log(D_\phi(\tilde{\mathbf{x}}_i)) - \sum_j \log(1 - D_\phi(\mathbf{y}_j))
$$

$$\tilde{\mathbf{x}}$$ 는 refine 된 합성 이미지이고, $$D_\phi$$ 는 이미지가 합성 이미지일 확률

### Local Adversarial Loss

Problem: 기존의 GAN으로 트레이닝 할 때는 Refiner가 특정 이미지 feature를 강조하여 discriminator를 속이려고 하는 경향이 있었음. 
  - 이 때문에 drifting이나 artifact가 생성됨.

이 논문에서는 합성 이미지를 여러개의 local patch로 나눴을 때, 각각의 local patch가 모두 실제 이미지의 patch와 비슷하게 되는 것을 목표로 함

Local patch로 나누면 discriminator의 receptive field와 capacity를 줄어들게 만들고, 이미지당 discriminator를 학습할 여러개의 sample을 얻는 효과를 얻을 수 있음

또한 refiner도 여러개의 realism loss를 가지게 되어 성능이 향상된다고 함

실제 구현할 때 adversarial loss은 모든 local patch들의 cross entropy loss 합

### Refined Image History

Problem: 기존의 GAN은 트레이닝 할 때 discriminator가 가장 최근에 생성된 이미지에만 집중
  - 이는 adversarial 트레이닝의 발산을 야기함
  - 또한 refine 네트워크가 discriminator가 까먹은 artifact를 다시 제시할 수 있게 함

이전의 refiner network로 생성한 이미지를 담은 buffer를 사용

discriminator를 학습할 때, mini-batch의 절반은 현재 refiner network로 만든 이미지, 나머지는 buffer에 있는 이미지를 사용해 학습

버퍼는 매 iteration마다 mini-batch의 크기의 절반만큼 랜덤으로 갱신시켜줌

[arXiv](https://arxiv.org/abs/1612.07828)
