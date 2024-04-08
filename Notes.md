# Makemore Notes

## Part-1 Bi-gram models

### Introduction

- We start by creating character level bigram models.
- Understand Pytorch Broadcasting.
- Need to play around a little with Tensors.
- Understand broadcasting rules in detail.

### Formulating a Deep Learning Problem

- Need to understand how to formulate a deep learning problem from a real-world problem
- Translate from real-world domain to deep-learning domain
- One example given in the bigram notebook is really good.
- You solve a problem using a probabilistic model
- You then, create an equivalent deep learning model which produces similar outputs.
- Learn the correlation, and techniques which transform the model space.

### Network Description

- Bigram network is a simple 1-layer FC network.

- Encode the input data using One-Hot-Encoding:
- You need to provide the right input as it will determine the activations which get triggered.

- Matrix Multiplication produces logits: which can be seen as log-counts.
- We use softmax at the end to create a probaiblity distribution.

- And then sample from the distribution during inference time.

### Formulating the loss

- For training we calculate the loss
- We want to maximise the likelihood of y-label of training data being selected during sampling from our model.
- But the probability is a number between 0-1 hences can diminish very quickly with subsequent multiplication over sequences
- Also from a deep learning perspective we want to deal with minimization of loss for optimization
  (Hopefully add some convexity to the problem)

- So we take the log of Maximum likelihood - log likelihood
- Now log for an input ranging from 0 to 1 is a -ve value
- We want to maximize this negative value
- Which is similar to saying minimise the negation of this value
- And that brings us to negative log likelihood being the loss that we try to minimise.
- You can average this over the batch size.

## Part-2 MLP

### Introduction

- RNNs are universal approximators; they are very expressive and in principle can implement all(questionable) algorithms.
- But they are not very easily optimizable using first - order techniques widely used.
- Key to understanding why they are not optimizable easily is by understanding:
  - the activations and gradients and how they behave during training.

### Curse of dimensionality:

- If we want to increase the sequence of inputs
  - The input space grows exponentially
  - for 2 character input sequence: 27 x 27 = 729
  - For 3 character input sequence: 729 x 27 ~ 20000
  - Also the counts for following characters become less and less
- So huge space and not a lot of information Summarizes Curse of dimensionality

- Few more points to consider:

  - Distance Concentration: Distance between nearest and the farthest point converges,
  - which complicates tasks like KNN search.
  - Increased Computational Complexity: Algorithms that work well on low-dimensional data become computationally infeasible in high dimensions due to increased data volume and the need for more complex models.
  - Overfitting: High Dimensions allow more opportunity to fit the noise.

- Base Paper:
  - Bengio et al (2003) Work on word level language model
  - But we will work with character level language modeling

### Motivation behind Mini-Batch

- Mini-Batch: Reduces the quality of gradient
- It is much better to have an approximate gradient and take more steps
- Than to evaluate the exact gradient and take fewer steps

### Finding the right learning rate

- Try some random values for the beginning learning rate
- See where the loss is exploding, such lrs (learning rate schedules) are outside the boundary
- For a range of lr that sufficiently reduces the loss and converges
  create a linspace over the range specifying appropriate intervals
- You can create the linspace over the exponents of the range as well
  In this case the actual lrs will be 10 \*\* exponents over the linspace.
- Now plot the graph of loss and lrs or lre and find the appropriate lr.
- At later stages to capture finer details you will have to decay the lr.

- This is roughly what we do

### Splitting the dataset

- 3 splits:
  - Training Split: Roughly 80% used to optimize the parameters of the model
  - Dev / Validation Split: Roughly 10% Development over the hyperparameters of the model
  - Test Split: Roughly 10% Evaluate the performance at the end. Use it very very sparingly

### Examples of Hyperparameters:

- Size of the Hidden Layer (Number of neurons in the hidden layer)
- Size of the embeddings
- Strength of regularization
- learning rate
- and more

### Obereservations and Subsequent modifications while training the network

- We observe a few things while training the network:

  - First:
    - There is a variance in the loss depending on the mini-batch inputs
    - Suggesting that the size of the mini-batch could be too low
    - We do not necessarily update the size of the mini-batches but we can take a look at this.
  - Second:
    - We also observe signs of underfitting or buggy behaviour as at times the trainig loss is larger than validation loss
    - Hence we try to increase the number of neurons in the hidden layer bumping it from 100 to 300.
  - Third:

    - At this point, another bottleneck that we observe is the dimension of the embedding space. with best train and val_loss achieved 2.30+
    - Increasing the dimension of the embedding space could provide higher expressive power for our model.

  - Final Improvements - On increasing the dimenions of the embeddings from 2 to 10
    and decreasing the hidden neurons number from 300 to 200 - We achieve a loss of around 2.20 showing better results.