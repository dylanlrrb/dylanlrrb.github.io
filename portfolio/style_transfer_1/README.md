<img src="./portfolio/style_transfer_1/assets/readme.png" width="700"/>

<img src="./portfolio/style_transfer_1/assets/readme3.png" width="700"/>

# Style Transfer with Backpropogation Through a Convolutional Neural Network

### Can we extarct and apply the abstractions learned by a deep neural network?


## Description

---

In this notebook I recreate a style transfer method outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

This is a realitivly slow method of style transfer; I will be exploring faster methods after this one!

Some questions that inevitably pop up when thinking about how to transfer the style from one image to the content of another image are, 
- How dow you quantify 'style'?
- How do you separate that separated 'style' from an image's 'content'?
- How do you apply that separated 'style' to another image's 'content'?

## Methods

---

The paper mentioned above explores the idea that content representations reside in higher layers in a trained convolutuional neural network (CNN) and style representations reside in lower levels in a trained CNN.

This makes intuitive sense as lower levels of a CNN learn to identify simple shapes, colors, edges, and so on while higher levels combime these simple features into more complex features (e.g. eyes, faces, dogs, houses, etc.) These complex features can be used to represent the content of an image while the lower level features can be used to represent style (e.g. colors, brushstrokes, motifs, etc.)

<img src="./portfolio/style_transfer_1/assets/readme2.png" width="600"/>

However, there is a bit more to 'style' that the paper explores. Style can be further described as low level features that are activated consistently together between the filter maps of a single low level layer in a CNN. In this way the 'style' of an image can be represented numerically by creating a gram matrix of a low level layer activations of a trianed CNN as an image is fed forward through it.

With these pieces defined, we can create a loss function that takes into account how different a target tensor's content layer activations are from a content image's activations, and how different a target tensor's style gram matrix is from a style image's style gram matrix.

Using this loss we can backpropogate the error through a network which has had it's parameters frozen to update the target tensor wich represents the final image!

## Results

---

While this method creates some amazing looking results, it is unfortunately rather slow.

Because it relies on backpropogation, many iterations of calculating the loss and backpropogating the error need to be preformed to get a good looking result. The speed also depends on the size of the image and if you have access to a GPU

Experimenting with some hyperparameters I found that cranking the learning rate up had a positive effect on how fast the target image converged to a good looking result. Additionally, I found that increasing how significantly the lower 'style' layers contribute to the loss calculation caused a more dramatic style transfer that required fewer interations to converge on a good looking result. And lastly, resizing the target to be a smaller tensor has a big impact on how fast the style can be transfered.




.