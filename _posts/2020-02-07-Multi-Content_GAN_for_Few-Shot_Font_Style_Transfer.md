---
title: "Multi-Content GAN for Few-Shot Font Style Transfer"
published: true
use_math: true
categories: 
  - review
---

Multi-Content GAN for Few-Shot Font Style Transfer (CVPR 2018)

--작성중--

기존의 conditional GAN을 하나만 사용해서 font를 생성하면 너무 많은 artifact가 생성됐었음

multi-content GAN 제시

먼저 전체 glyph(글자) 모양을 모델링한 후, color와 texture를 입혀주는 방식

## Multi-Content GAN Architecture

각각의 network 구조는 conditional GAN(cGAN) 를 조금 수정해서 만듦

### cGAN

기존의 GAN에서는 랜덤한 noise vector $z$로 이미지 $y$를 생성하는게 목표 ($z → y$)

cGAN에서는 추가적으로 conditional image인 $x$를 추가해줌 ($$\{x, z\} → y$$)

따라서 loss function은 아래처럼 된다

$$
\mathcal{L}_{\textrm{cGAN}}(G,D) = \mathbb{E}_{x,y\sim p_{\textrm{data}}(x,y)}[\log D(x,y)] \\
+ \mathbb{E}_{x\sim p_{\textrm{data}}(x),z\sim p_z(z)}[1 - \log D(x,G(x,z))]
$$

Generator의 output에 대한 GT가 주어져 있을땐 Generator의 타깃에 대한 L1 loss를 추가해주면 더 좋다고 한다

$$
G^* = \arg \min_G \max_D \mathcal{L}_{\textrm{cGAN}}(G,D) + \lambda \mathcal{L}_{L_1}(G)
$$

아래의 네트워크들은 이 cGAN 구조를 따랐으며, [Image-to-Image translation](https://arxiv.org/abs/1611.07004) 처럼 random noise $z$는 사용하지 않았다고 한다

### GlyphNet

<figure class="align-center">
  <img src="/assets/images/post/200207/GlyphNet.png" alt="">
  <center><figcaption>GlyphNet Architecture</figcaption></center>
</figure> 

GlyphNet은 26개의 각 알파벳끼리의 상관관계(correlation)과 유사도를 학습함

입력값의 사이즈는 $B \times 26 \times 64 \times 64$, $B$는 Batch size 이다

Discriminator는 PatchGAN에서 쓰였던 것처럼 $21 \times 21$ 사이즈의 local discriminator를 사용하였음 (Convolution layer 3개짜리)

추가로 global discriminator 사용 (Convolution layer 2개짜리)

loss function은 LSGAN loss에 추가로 L1 loss 사용

참고로 GlyphNet에서 사용한 discriminator는 GlyphNet을 pretrain할 때에만 사용하고, OrnaNet과 합친 최종 end-to-end 모델에서는 사용하지 않는다고 함

학습할 때 입력값 $x_1$는 랜덤하게 정해진 몇몇 알파벳을 제외한 다른 알파벳들은 모두 0인 벡터

### OrnaNet

OrnaNet은 GlyphNet이 gray-scale font $x_2$를 제공하면, 그 이미지를 바탕으로 채색과 그 외 꾸밈(ornamentation)을 해준다

다루는 데이터의 차원, 타입, 그리고 얼마나 모델이 이미지를 만드는 데 있어서 구체적인지를 제외하면 전체적인 구조는 GlyphNet과 완전히 동일하다

### End-to-End Network

<figure class="align-center">
  <img src="/assets/images/post/200207/end_to_end_architecture.png" alt="">
  <center><figcaption>End-to-End Architecture</figcaption></center>
</figure> 

End-to-End Network가 작동하는 과정은 다음과 같다

1. 랜덤한 하나의 단어 (위 그림에서는 $TOWER$)에서 한 글자씩을 제외한 후, GlyphNet에 넣는다
2. GlyphNet이 1에서 제외한 글자를 예측해서 생성한다 
3. 이런식으로 26개의 알파벳을 모두 예측한다 (주어진 단어에 없는 알파벳은 단어를 전부 넣어서 예측함)
4. 생성된 $1 \times 26 \times 64 \times 64$ 크기의 벡터를(Glyph Stack) OrnaNet에 전달하여 꾸며줌

또한 OrnaNet에서 깨끗한 outline의 이미지를 생성해내기 위해 입력값($x_2$)과 출력값의 binary mask 사이의 MSE를 loss에 추가해 주었다고 함

이 때, binary mask는 이미지에 sigmoid 함수를 적용해 만들어냈다고 함 (그림에서 $\sigma$)

최종적으로 OrnaNet의 loss function은 다음과 같다

$$
\mathcal{L}(G_2) = \mathcal{L}_{\textrm{LSGAN}}(G_2, D_2) + \lambda_1 \mathcal{L}_{L_1}(G_2) + \lambda_2 \mathcal{L}_{\textrm{MSE}}(G_2)
$$

## Datasets

학습엔 직접 만든 font dataset을 사용했다고 한다

1만개의 gray-scale 라틴 대문자 (26자) 폰트를 $64 \times 64$ 크기로 resize 해줬음

또한 각 font 마다 두 개의 랜덤한 color gradient와 outline을 적용해 총 2만개의 color font 데이터셋을 만들었다고 함