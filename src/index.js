projects = [
  "mnist_nn.js",
  "style_transfer_1.js",
  "backprop_painting.js",
  "docker_tutorial.js",
  "supervised_learning.js",
  "unsupervised_learning.js",
  "signal_separation.js",
  "background_removal.js",
  "cifar10_classification.js",
  "landmark_classification.js",
  "lstm_scripts.js",
  "movie_sentiment.js",
  "denoise_autoencoder.js",
  "word2vec.js",
  "cat_dog_classifier.js",
  // genomics PCA
  // View feature maps in a real time web app 
  // Multi box detector using mobile net + demo app (first expor`ation in pytorch then implementationin a web app)
  // Class activation Map/ Saliency Map/ Attribution Maps
  // Deep dream clone
  // tyle transfer pt 2, model per style to transfer 1
  // style transfter pt 3, arbitrary style transfer in browser (DEMO)
  // project with seq2seq without attention (translation in order to compare to w/ attention?)
  // Translation with Attention project (seq2seq w/ attention) + class activation map of attention matrix as sentence is translated 1, 2, 3, 4
  // Image captioning with attention, visualizing focused parts of image attention is given to 1
  // Create images from text
  // Add color to a BW image
  // Image super resolution auto encoder
  // Compare dimensionality reduction with autoencoders vs PCA
  // brainwave signal separation and deep learning 
].map((x) => `./src/projects/${x}`)

demos = [
  "mnist_nn.js",
  "docker_tutorial.js",
  "cat_dog_classifier.js",
  // View feature maps in a real time web app
  // Multi box detector using mobile net + demo app (first exporation in pytorch then implementationin a web app)
  // style transfter pt 3, arbitrary style transfer in browser (DEMO) 
].map((x) => `./src/demos/${x}`)

placeholders = [
  "00_tile_placeholder.js",
  "00_tile_placeholder.js",
  "00_tile_placeholder.js",
].map((x) => `./src/projects/${x}`)
            

function loadScript(src) {
  return () => {
    const script = document.createElement('script')
    script.src = src
    document.body.append(script)
    return new Promise((res, rej) => {
      script.onload = res
      script.onerror = rej
    })
  }
}

function sequential(p) {
  i = 0
  recurse = () =>  {p[i]().then(() => {i++;if (i >= p.length) {return} else {recurse()}})}
  recurse()
}

window.addEventListener('DOMContentLoaded', () => {
  const hamburger_icon = document.querySelector('header.mobile .hamburger-icon')
  const mobile_nav = document.querySelector('header.mobile .nav')
  const selfie = document.querySelector('div.selfie')

  hamburger_icon.addEventListener('click', () => {
    mobile_nav.classList.toggle('closed')
    document.documentElement.classList.toggle('disable-scroll')
  })
  mobile_nav.addEventListener('click', (e) => {
    if (!e.target.classList.contains('nav')) {
      mobile_nav.classList.toggle('closed')
      document.documentElement.classList.toggle('disable-scroll')
    }
  })
  Array.from(document.querySelectorAll('section')).forEach((section) => {
    section.addEventListener('touchstart', () => {
      mobile_nav.classList.add('closed')
      document.documentElement.classList.remove('disable-scroll')
    })
  })

  const node = document.createElement('div')
  node.classList.add('transparent')
  node.classList.add('high')
  setTimeout(() => {
    selfie.appendChild(node)
  }, 100)
  setTimeout(() => node.classList.toggle('transparent'), 300)

  loadScript('https://cdnjs.cloudflare.com/ajax/libs/showdown/2.0.0/showdown.min.js')()
    .then(() => loadScript('./src/helpers/render_tiles.js')())
    .then(() => sequential([...projects, ...placeholders, ...demos, './src/projects/00_remove_loader.js'].map(loadScript)))
    .catch(() => console.log('error loading showdown script'))
  
});
