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

## Part-3 Activations, Gradients and BatchNorms

Loss-Graph looks like a hockey stick because not good initializations resulting in some predictions which are wrong being too confidently predicted.
Initialization is important as when we set up our network the network still hasn't learned anything.
In such a scenario the output it generates should be uniform to a large extent over the output space.

Too confident of a wrong prediction can lead to the loss being large
which leads to gradients being large
which leads to weight updates being large
which doesn't lead to a smooth optimization of model parameters

If you fix this, you get a better loss at the end of training.
As it is using training more efficiently not wasting the time to
squash the weights to the right values: Hockey stick nature of loss over iterations, the stick is not that useful (Could be a few thousand iterations)

### Activations

- We want to control the activations
- Shouldn't be squashed to zero or explode to infeasible values
- We want a roughly gaussian activation
- How do we scale the weights and biases so that everything is as controlled as possible (kaiming initialization)

### Normalization

- Batch Normalization: Probably came out first
- A layer which can be sprinkled throughout the net wherever linearity is present. Before activations.
- It leads to Centering of the inputs to the next layer by coupling of all examples
- Due to this centering effect, individual biases loose the value and hence we require two new parameters for the batchNorm layer
- bngain and bnbias
- Our final equation on the output of the batchnorm layer looks like this: bngain \* (batch normalization) + bnBias

Normalization Causes huge amounts of bugs, becuase it couples examples vertically in the neural net
Avoid it as much as possible, but seems to be super powerful

Also intrinsically provides regularization

Part 3 also shows how to analyse the behaviour of the network by studying the characteristics of your activations and gradients being propagated. You want to ensure that all the layers receive values whch have similar distribution. This is discussed well in the video.

## Part-4 Becoming a BackProp Ninja

### Introduction

We essentially write the backward pass for a network manually instead of using Pytorch's loss.backward()
Similar to what we did for micrograd.
The difference here is that we are dealing with vectors instead of scalars.

### Important Observations

Always remember, the end goal is to calculate the gradient of loss with respect to each parameter that has contributed to the loss.
Hence, only the parameters that have CONTRIBUTED to the accumulation of loss will get affected.
This makes finding the derivatives a lot more intuitive.

Rely on dimensionality matching to compute the gradients.
Dimensions of gradients for a parameter = Dimensions of parameter

If there is squashing in the forward pass, we will need broadcasting in the backward.
If there is broadcasting in the forward pass, we will need squashing in the backward.
All of this stems from dimensionality matching.

If a parameter has multiple fan_outs:

- Its gradient will be an accumulation of all of its local derivatives with subsequent parameters multiplied by their respective gradients.

### Basic gradient calculation example

```
g = f(x)
if dg is calculated.
dx can be calculated using the local derivative and the incoming gradient.
dg === incoming gradient === dL/dg
dL/dx = dg/dx \* dL/dg
```

Based on this nature of backpropagation:
We need to ensure that all the incoming gradients are calculated before we calculate the gradient for a parameter.

We can club huge chunks of network into a single pass as long as we are able to effectively compute its gradients.
This can lead to boost in efficiency as lesser compute is required due to reductino in intermediary layers from input to output.

For this, you will have to express multiple passes as a single composite function and analytically derive the gradient.

### Bessel's correction

Bessel's correction: denominator should be (n - 1) for variance as it is a more realistic approximation as compared to (n).
Doesn't matter so much if the value of n is large.
But let's take the example of Batch Normalization:
We can have sufficiently small size of batches and in such case it is better to stick with the Bessel's correction.

### TODO

Practise analytical calculation of gradients for composite functions. A couple available in Part4 vide.

## Wavenet and Designing networks for more efficient gpu compute utilization.

Wavenet processes sequences of audio data using dilated convolutions to capture temporal dependencies.
Torch matmul implementation and other functions:
It only matches the dimensions of the very last value in shape of the first tensor with the very first value of shape in the second tensor while using @

We need to be careful while implementing different layers as they should also be able to process this multidimensional input.
Specifically the BatchNorm Layer needs to ensure that it is only calculating the mean and variance along the desired dimensions.
We handle that in this lecture.

Haven't implemented the residual connections and skip connections

### Why convolutions: Makes it more efficient to train the network

- Doesn't change the behavioiur of the network
- Convolutions allows us to slide the entire model efficiently over the input sequence.
- Makes the for loop: inside the cuda and not in python.

### Notes on Development Process

- Prototype in Jupyter and once satisfied with the functionality
- Copy paste into the main repository
- Kick off experiements.
