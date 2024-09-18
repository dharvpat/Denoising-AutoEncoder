# Real-Time Denoising AutoEncoder Using PyTorch's MPS on Mac

## Introduction

This project implements a real-time Denoising AutoEncoder (DAE) using PyTorch with MPS backend on Mac for GPU acceleration.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- macOS with MPS-compatible GPU (Apple Silicon)

### References


Haykin, S. (2002). Adaptive Filter Theory. Prentice Hall.

Kuo, S. M., & Morgan, D. R. (1996). Active Noise Control Systems: Algorithms and DSP Implementations. Wiley.

Pascual, S., Bonafonte, A., & Serra, J. (2017). SEGAN: Speech Enhancement Generative Adversarial Network. arXiv preprint arXiv:1703.09452.

Fu, S.-W., Tsao, Y., Lu, X., & Hwang, K. (2017). End-to-End Waveform Utterance Enhancement for Direct Evaluation Metrics Optimization by Fully Convolutional Neural Networks. arXiv preprint arXiv:1709.03658.

Valentini-Botinhao, C., Wang, X., Takaki, S., & Yamagishi, J. (2016). Investigating RNN-based Speech Enhancement Methods for Noise-Robust Text-to-Speech. In 9th ISCA Speech Synthesis Workshop (pp. 146-152).

Reddy, C. K., et al. (2020). The Interspeech 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results. Interspeech 2020.

Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.

Goodfellow, I., et al. (2014). Generative Adversarial Nets. Advances in Neural Information Processing Systems, 27.

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. Science, 313(5786), 504-507.

Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and Composing Robust Features with Denoising Autoencoders. Proceedings of the 25th International Conference on Machine Learning, 1096-1103.

#### Descriptions of papers referenced

Haykin (2002) provides a comprehensive overview of adaptive filter theory, essential for understanding traditional noise cancellation techniques.

Kuo & Morgan (1996) delve into active noise control systems and digital signal processing implementations, offering foundational knowledge for noise cancellation projects.

Pascual et al. (2017) introduce SEGAN, a generative adversarial network for speech enhancement, demonstrating the application of GANs in denoising tasks.

Fu et al. (2017) explore end-to-end waveform enhancement using fully convolutional neural networks, highlighting deep learning approaches to audio denoising.

Valentini-Botinhao et al. (2016) present the VoiceBank-DEMAND dataset and investigate RNN-based speech enhancement methods, contributing valuable resources and insights for model training.

Reddy et al. (2020) discuss the Deep Noise Suppression Challenge and provide datasets and evaluation frameworks, crucial for benchmarking noise suppression models.

Kingma & Welling (2014) propose the variational autoencoder, a generative model that combines neural networks with variational inference, influencing autoencoder architectures.

Goodfellow et al. (2014) introduce generative adversarial networks (GANs), a framework that has significantly impacted generative modeling and denoising applications.

Hinton & Salakhutdinov (2006) demonstrate how neural networks can reduce data dimensionality, foundational for understanding autoencoders.

Vincent et al. (2008) present denoising autoencoders, showing how they can learn robust data representations, directly applicable to this project's objectives.