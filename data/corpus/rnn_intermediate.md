# Recurrent Neural Networks (RNNs)

## Overview

A Recurrent Neural Network is a type of neural network designed to process sequential data. Unlike feedforward networks, which treat each input independently, RNNs maintain a hidden state that evolves over time and carries information from previous inputs into future ones. This makes them useful for tasks involving time series, language, audio, and other ordered data.

At each time step, an RNN takes an input vector and combines it with the previous hidden state to produce a new hidden state. The recurrence creates a memory of what has happened earlier in the sequence.

## Why RNNs Matter

Many real-world tasks depend on context, not just isolated observations. For example:

- In natural language processing, the meaning of a word depends on the preceding words.
- In stock prediction, future values may depend on earlier trends.
- In speech recognition, sounds must be interpreted in the proper temporal order.

RNNs are naturally suited to these sequence tasks because they preserve information over time.

## Hidden State and Recurrence

An RNN computes a hidden state using a recurrent formula such as:

- `h_t = f(W_x x_t + W_h h_{t-1} + b)`

Here:

- `x_t` is the current input
- `h_{t-1}` is the previous hidden state
- `W_x` and `W_h` are learned weight matrices
- `f` is a nonlinear activation function such as tanh or ReLU

This recurrence allows the model to summarize past information within its internal memory. The output at each step may be a prediction, a hidden representation, or part of a sequence-to-sequence model.

## Backpropagation Through Time

Training an RNN uses a method called Backpropagation Through Time (BPTT). The idea is to unroll the network across time steps and apply standard backpropagation to the resulting chain of computations. This enables gradients to flow through the sequence and adjust the weights accordingly.

The challenge is that the same weights are reused at every step, which can cause gradients to become extremely small or very large over long sequences. This is known as the vanishing or exploding gradient problem.

## Vanishing Gradient Problem

The vanishing gradient problem is one of the main reasons plain RNNs struggle on long sequences. When the network is unfolded over many time steps, repeated multiplication by weights can shrink gradients until they vanish. As a result, the model learns poorly from distant time dependencies.

This is why early RNNs were weak at long-range reasoning, such as understanding a sentence where the key clue appears many words earlier.

## LSTM and GRU

To address vanishing gradients, researchers introduced specialized RNN variants such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks.

- LSTMs use gates that control what information to keep, forget, and output.
- GRUs simplify this design with fewer gates while retaining much of the performance.

These models are better at representing long-term dependencies and have become standard for sequence modeling tasks.

## Common Applications

RNNs and their variants are commonly used in:

- Language modeling
- Machine translation
- Sentiment analysis
- Speech recognition
- Time series forecasting
- Video frame prediction

## Interview Angle

A good interview answer should explain that RNNs are sequence models with shared parameters over time, and that their recurrence gives them memory. You should also mention the training challenge of long-range dependencies and the fact that LSTMs/GRUs were introduced specifically to mitigate vanishing gradients.

## Key Takeaway

RNNs are foundational sequence models because they maintain a hidden state that captures temporal context. While effective for short patterns and many classic sequence tasks, they can struggle with long-term dependencies unless enhanced using LSTM or GRU architectures.
