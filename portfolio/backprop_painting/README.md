<img src="./portfolio/backprop_painting/assets/readme.jpeg" width="500"/>

# Visualizing Convolutional Layers in a Trained VGG Network

### What have the various feature maps in a CNN been trianed to look for?

## Description

---

Convolutional Neural Nets (CNNs) are very deep data sructutes that extract increasingly complex spacial patterns in 2d or 3d data.

A CNN trained on a large corpus of images will build representations of the training at different levels of complexity at different layers of the network. In theory, lower layers will build representations of low level features such as colors and edges. Then higer levels will combine thos representations into increasingly complex feature detectors, e.g. edges -> corners -> squares -> doors -> houses

This ability for CNNs to find patterns within patterns within patterns makes them extrmely powerful, but also extremly mysterious. Is there a way to increase the interpretability of what CNNs are doing as an image is fed forward through the network?

## Methods

---

In this exploration, I will be using a pretrianed VGG16 network wo build an image that most intensly 'excites' a subset of feature maps in specific convolutional layers.

The convolutional layers I will be examining are:
- Conv1_1 with 64 feature maps to explore
- Conv2_1 with 128 feature maps to explore
- Conv3_1 with 256 feature maps to explore
- Conv4_1 with 512 feature maps to explore
- Conv5_1 with 512 feature maps to explore

<img src="./portfolio/backprop_painting/assets/readme_a.png" width="600"/>

I do this by starting with a targer tensor of random noise and feeding it through the pretrained VGG network which has had it's parameters frozen. At a specific feature map in the layer I am examining, I grab the activations calculate a loss by comparing the actual activation tensor to a target activation that is unreachably high (In this case a target tensor filled with the maximum squared value in the actual tensor). I then backpropogate the loss to update the target tensor; In this way after each iteration it continually updates into an input that maximally actucated the feature map in question.

I then display the tensor as an image for an idea of what that particualr feature map has been trianed to look for! 

## Results

---

Overall the results are quite impressive. You can obseve that in lower levels such as Conv1_1 feature maps are activated by relitivly solid colors and simple lines and shapes. I can deduce that this layer's feature maps have, over the course of training on the images dataset, learned to become simple color and edge detectors.

This is in contrast to the highest layers which are activated by complicated and abstract patterns. These generated images are neat to look at, but some have enough form to actually guess what high level feature they are detecting!
For exapmle, I found the 25th feature map of the Conv5_1 layer to be excited specifically by patterns that resemble human eyes! do you see any eyes staring back at you from this generated image?

<img src="./portfolio/backprop_painting/assets/output1.png" width="250"/>

I can also see what resemble animal eyes in the 116th feature map of the Conv5-1 layer and what strike me as perhaps birds/parakeets in the 104th feature map of the Conv5-1 layer

<img src="./portfolio/backprop_painting/assets/output2.jpeg" width="500"/>


You can also observe blank and noisy squares generated for some feature maps, which appear to be more frequent in higher layers.
At first I thought that it might mean that feature map did not actually learn anything useful during training, but i think the reality is that this method of backpropogation used is sensitive to starting conditions. I suspect that higher layers have a harder time finding a complex pattern that excites thm in the noisy starting tensor, backprop can't get a 'footing', and never converge to anything meaningful.




.