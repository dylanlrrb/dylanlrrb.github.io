<!-- <img src="./portfolio/landmark_classification/assets/readme.png" width="400"/> -->

<img src="./portfolio/landmark_classification/assets/atomium_input.png" width=250>

### Input Image

<img src="./portfolio/landmark_classification/assets/transfer_atomium.png" width=800>

### Visually Similar Images



# Implementing Reverse Image Search with Landmark Images

### Using a CNN to encode images into searchable semeantic vectors

## Description

---

In this project I explore using convolutional Neural Networks (CNNs) to encode an image into a latent space vector. This method can then be used to index those images to be searched against the vector of an input image to find visually similar images.

## Methods

---

I use images of landmarks in this project, and 3 different CNN archetectures to compare results

The 3 CNN archetectures that I use are:
- a made from scratch CNN with 4 convolutional layers and a classifier made of 3 fully connected layers, trained on the landmark image dataset
- an unmodified, pre-trained VGG16 network
- a VGG16 network with transfer learning, trained on the landmak image dataset

After training each model (if necessary) I use TSNE with 2 components to visualize the closeness of image vectors in 2D space of the 50 different landmark classes in the dataset. Classes are identified by the same colors and labled with thier class number, so clusters of classes indicate those image vectors are similar and it be feisable to use euclidian distance to determine visual similarity 

I then use a function I wrote to rank each images latent space vector aganst the vector of an input image to display visually similar images.

## Results

---

Overall this method of using vector similarity to lookup visually similar images works quite well. Additionally each model archetecture used for encoding the images have somewhat different properties and possible use cases.

---

### From Scratch CNN

While the scratch CNN architecture found a few images with the correct class, all the images share colors and textures with the input image. Looking a the TSNE for images encoded by this model, I can observe some loose clustering of classes. In my opinion, the top results all share the same "mood" and could find use in finding images that stylistically go well together, rather than share the same subject as the input image. This is supported by the TSNE obsevation wich shows significant overlap and spread of clusters.

<img src="./portfolio/landmark_classification/assets/scratch_tsne.png" width=300>

### TSNE on images encoded with from-scratch CNN

<img src="./portfolio/landmark_classification/assets/soreq_input.png" width=250>

### Input Image

<img src="./portfolio/landmark_classification/assets/scratch_soreq.png" width=800>

### Visually Similar Images

---

### Pre-trained VGG network

The pre-trained VGG does better than the from-scratch CNN archetecture at identifying images that share the same subject as the input image; and the ones that it does image share colors, textures and composition with the input image. Looking at the TSNE, there are some loose clusters that appear slightly more condensed that the from-scratch TSNE, supports the observation that it preforms better than from-scratch model at identifying images of similar classes. The pre-trained VGG has the advantage of working out of the box - it does not need any fine tuning or trnasfer learning to work decently, making this approach suitable for a general reverse image search application where a specific domain cannot be trained for ahead of time.

<img src="./portfolio/landmark_classification/assets/vgg_tsne.png" width=300>

### TSNE on images encoded with pre-trained VGG

<img src="./portfolio/landmark_classification/assets/soreq_input.png" width=250>

### Input Image

<img src="./portfolio/landmark_classification/assets/vgg_soreq.png" width=800>

### Visually Similar Images

---

### VGG Network with Transfer Learning

The most prominent observation from the transfer learning model is hos closely clustered most classes are in the TSNE. As a result, the reverse image search returns images of all the same class as the input images. This method of encoding has a very high accuracy but the tradeoff is that the subject domain of the images would need to be known ahead of time, a dataset needs to exist for it, and it needs to be trained with it. Where the vanilla pre-trained model could have use for general reverse image search use cases, transfer learning models could be used for specific domain areas. For example looking up shoes, or plants, or (in this case) landmarks.

<img src="./portfolio/landmark_classification/assets/transfer_tsne.png" width=300>

### TSNE on images encoded with VGG with transfer learning

<img src="./portfolio/landmark_classification/assets/soreq_input.png" width=250>

### Input Image

<img src="./portfolio/landmark_classification/assets/transfer_soreq.png" width=800>

### Visually Similar Images

<br><br><br><br>