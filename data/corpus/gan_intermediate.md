# Generative Adversarial Networks (GANs)

## Overview

A Generative Adversarial Network, or GAN, is a neural architecture designed to learn a data distribution and generate new samples that resemble it. Instead of simply fitting a probability density, GANs train two competing networks:

- The generator creates synthetic samples.
- The discriminator tries to distinguish real samples from generated ones.

This adversarial setup is powerful because the generator does not receive a direct target output; instead, it improves by learning how to fool the discriminator. Over time, the generator becomes better at producing realistic samples, while the discriminator becomes better at spotting fakes.

## Core Idea

GANs are inspired by a zero-sum game. The generator produces a sample `G(z)` from random noise `z`, and the discriminator evaluates whether the sample is real or fake. The objective is usually written as a minimax problem:

- The generator tries to minimize the chance that the discriminator correctly identifies fake data.
- The discriminator tries to maximize its ability to correctly label real and generated samples.

In practice, the discriminator outputs a probability that a sample is real, and the generator is updated through gradients flowing from the discriminator back to the generator.

## Generator and Discriminator

The generator often starts from a low-dimensional latent vector, such as a random noise vector in 100 or 512 dimensions. It transforms this vector through several layers, often transposed convolutions, into an image or other high-dimensional output. The discriminator, in contrast, takes a real or generated sample and predicts whether it came from the true data distribution.

A good discriminator learns the structure of the data manifold, while the generator learns to place points convincingly on that manifold. This is why GANs are especially effective for image synthesis, style transfer, super-resolution, and data augmentation.

## Why Training Is Difficult

GANs are notoriously unstable. Common problems include:

- Mode collapse, where the generator produces only a small set of similar outputs.
- Oscillation, where the generator and discriminator keep chasing each other without converging.
- Vanishing gradients, which happen when the discriminator becomes too strong too early and the generator receives weak learning signals.

Practical techniques to stabilize training include minibatch discrimination, label smoothing, feature matching, and better architectural choices such as batch normalization or Wasserstein distance.

## Wasserstein GANs

A major improvement came with Wasserstein GANs (WGANs). Rather than using the original Jensen-Shannon divergence, WGANs use the Wasserstein distance to measure how different the real and generated distributions are. This often makes training more stable and reduces issues like mode collapse.

A WGAN typically adds a critic instead of a discriminator and applies a Lipschitz constraint, often through weight clipping or gradient penalty. This encourages smoother learning dynamics and has become a standard choice in many modern generative pipelines.

## Applications

GANs are used for:

- Image generation and synthesis
- Photo restoration and super-resolution
- Domain adaptation and style transfer
- Data augmentation in imbalanced datasets
- Anomaly detection in some tasks
- Generative art and simulation

## Interview Angle

A strong answer to a GAN interview question should explain the adversarial objective, the roles of generator and discriminator, and why training is harder than in standard supervised learning. You should also mention the tradeoff between sample quality and training stability, since GANs create impressive outputs but are often difficult to tune reliably.

## Key Takeaway

GANs are a powerful class of generative models that learn through competition rather than direct supervision. They are capable of producing highly realistic synthetic samples, but they demand careful engineering to manage instability and optimization challenges.
