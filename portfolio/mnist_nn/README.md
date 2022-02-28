<img src="./portfolio/mnist_nn/assets/readme.gif" alt="app_image" width="300"/>

# Real Time Activation Visualization of Hidden Layers in a Network that Classifies Handwritten Digits

### See what a Neural Network is thinkning as it guesses what number you're drawing


## Description

---

This demo allows a user to draw digits and see how activations in the hidden and output layers change in real time.

It can be difficult to visualize exactly what a individual neurons are doing as inputs are fed forward through a neural network.

I build this Demo as a way for myself and others to better understand the weights and biases in a network as well as how activations at various layers change in real time as an input changes.

The term "hidden layer" when refering to neural networks often sounds like a mysterious black box. By creating a visual understanding of what the math used by a multilayer perceptron represents, I attempt to de-mystify some of the abstract and difficult to grasp concepts of neural networks

<img src="./portfolio/mnist_nn/assets/readme2.jpeg" width="250"/>

Begin by drawing a digit in the red boxed area, of the gray grid; classification works best if the digit fills the red boxed area.

click on a neuron in the hidden layer to see the weights of how it is connected to the input area

The same dataset was triained with vaious regualrization methods and cost functionswhich you can toggle between to see the effect on the network weights.

For example, comparing L1 vs. L2 regualrization and observing the difference in input layer weights really drives home the differnce in weight sparsity between the two method sin a very visual an intuitive way

