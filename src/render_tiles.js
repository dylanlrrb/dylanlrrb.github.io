const renderProject = ({tileImage, tileTitle, markdownUrl, controls}) => {
  const html = `<div class="project">
    <div class="tile">
      <div class="tile-image"></div>
      <p class="tile-title">${tileTitle}</p>
    </div>
    <div class="modal display-none">
      <div class="modal-scrim"></div>
      <div class="modal-content">
        <div class="modal-header">
          <img class="exit exit-button" src="public/icons/x_icon.svg" alt="Open menu">
        </div>
        <div class="modal-inner"></div>
        <div class="modal-fade"></div>
        <div class="modal-controls">
          ${controls.join('')}
        </div>
      </div>
    </div>
  </div>`
  const node = document.createRange().createContextualFragment(html);
  const tile = node.querySelector('.tile')
  const tileIm = node.querySelector('.tile-image')
  const modal = node.querySelector('.modal')
  const exit = modal.querySelector('.exit')
  const scrim = modal.querySelector('.modal-scrim')
  const inner = modal.querySelector('.modal-inner')
  
  return () => {
    document.querySelector("#projects").append(node)
    tile.addEventListener('click', () => {
      modal.classList.toggle('display-none')
      document.documentElement.classList.add('disable-scroll')
      window.history.pushState({modal: 'open'}, '', window.location)
    })
    window.addEventListener("popstate", function () {
      modal.classList.add('display-none')
      document.documentElement.classList.remove('disable-scroll')
    }, true)
    exit.addEventListener('click', () => window.history.back())
    scrim.addEventListener('click', () => window.history.back())
    setTimeout(() => {
      tileIm.style['background-image'] = `url(${tileImage})`
      const converter = new showdown.Converter({strikethrough:true, tables:true, tasklists:true, emoji:true, openLinksInNewWindow:true})
      fetch(markdownUrl).then(res => res.text())
        .then(text => {
          inner.innerHTML = converter.makeHtml(text)
          document.querySelector("#projects").append(node)
        })
    }, 1000)
  }
}

const renderDemo = (demo) => {
  const {demoImage, demoDescription, demoLink} = demo
  const fragment = `<a href="${demoLink}" target="_blank" class="demo-tile">
                      <div class="demo-image"></div>
                      <div class="demo-description">${demoDescription}</div>
                    </a>`
  const node = document.createRange().createContextualFragment(fragment)
  const demoIm = node.querySelector('.demo-image')

  return () => {
    document.querySelector("#demo-list").append(node)
    setTimeout(() => {
      demoIm.style.cssText = `background-image:linear-gradient(to right, transparent, transparent, white), url(${demoImage});
        background-image:-o-linear-gradient(to right, transparent, transparent, white), url(${demoImage});
        background-image:-webkit-gradient(to right, transparent, transparent, white), url(${demoImage});`
    }, 1000)
  }
}


// portfolio

window.projects = {
  placeholder : () => {
    const node = document.createRange().createContextualFragment(`<div class="tile-placeholder"></div>`);
    document.querySelector("#projects").append(node)
  },

  remove_loader: () => {
    document.querySelector('.loading-scrim').style['display'] = 'none'
  },

  background_removal: renderProject({
    tileImage: './portfolio/background_removal/assets/portfolio_tile.gif',
    tileTitle: 'Using Gaussian Mixture Models to Isolate Movement in Video',
    markdownUrl: './portfolio/background_removal/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/background_removal/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  backprop_painting: renderProject({
    tileImage: './portfolio/backprop_painting/assets/portfolio_tile.png',
    tileTitle: 'Visualizing Convolutional Layers in a Trained VGG Network',
    markdownUrl: './portfolio/backprop_painting/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/backprop_painting/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  cat_dog_classifier: renderProject({
    tileImage: './portfolio/cat_dog_classifier/assets/portfolio_tile.png',
    tileTitle: 'Classifying Cat Vs. Dog Images with a MobileNet',
    markdownUrl: './portfolio/cat_dog_classifier/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/cat_dog_classifier/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  cifar10_classification: renderProject({
    tileImage: './portfolio/cifar10_classification/assets/portfolio_tile.png',
    tileTitle: 'Transfer Learning with the CIFAR-10 dataset',
    markdownUrl: './portfolio/cifar10_classification/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/cifar10_classification/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  denoise_autoencoder: renderProject({
    tileImage: './portfolio/denoise_autoencoder/assets/portfolio_tile.png',
    tileTitle: 'Using Autoencoders for Image Noise Reduction',
    markdownUrl: './portfolio/denoise_autoencoder/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/denoise_autoencoder/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  docker_tutorial: renderProject({
    tileImage: './portfolio/docker_tutorial/assets/portfolio_tile.png',
    tileTitle: 'Project-Based Docker Tutorial',
    markdownUrl: 'https://raw.githubusercontent.com/dylanlrrb/Please-Contain-Yourself/master/README.md',
    controls: [
      `<a href="https://github.com/dylanlrrb/Please-Contain-Yourself" target="_blank">
        <button class="primary-action-button">Tutorial</button></a>`,
    ],
  }),

  landmark_classification: renderProject({
    tileImage: './portfolio/landmark_classification/assets/portfolio_tile.png',
    tileTitle: 'TSNE analysis on VGG Hidden Layer Activations of Landmark Images',
    markdownUrl: './portfolio/landmark_classification/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/landmark_classification/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  lstm_scripts: renderProject({
    tileImage: './portfolio/lstm_scripts/assets/portfolio_tile.png',
    tileTitle: 'Generating TV Scripts with LSTM Recurrent Neural Networks',
    markdownUrl: './portfolio/lstm_scripts/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/lstm_scripts/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  mnist_nn: renderProject({
    tileImage: './portfolio/mnist_nn/assets/portfolio_tile.gif',
    tileTitle: 'Real Time Activation and Weight Visualization of Neural Network Trained on MNIST Dataset',
    markdownUrl: './portfolio/mnist_nn/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/neural-networks-from-scratch/blob/main/network2.ipynb" target="_blank">
        <button class="secondary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/mnist_nn/index.html" target="_blank"><button class="primary-action-button">Demo</button></a>`
    ],
  }),

  movie_sentiment: renderProject({
    tileImage: './portfolio/movie_sentiment/assets/portfolio_tile.png',
    tileTitle: 'Sentiment Analysis of Movie Reviews with LSTM Recurrent Neural Nets',
    markdownUrl: './portfolio/movie_sentiment/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/movie_sentiment/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  signal_separation: renderProject({
    tileImage: './portfolio/signal_separation/assets/portfolio_tile.png',
    tileTitle: 'Blind Source Signal Separation',
    markdownUrl: './portfolio/signal_separation/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/signal_separation/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  style_transfer_1: renderProject({
    tileImage: './portfolio/style_transfer_1/assets/portfolio_tile.gif',
    tileTitle: 'Style Transfer with Backpropogation Through a Convolutional Neural Network',
    markdownUrl: './portfolio/style_transfer_1/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/style_transfer_1/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  supervised_learning: renderProject({
    tileImage: './portfolio/supervised_learning/assets/portfolio_tile.png',
    tileTitle: 'Supervised Learning Methods Comparison Using Census Data',
    markdownUrl: './portfolio/supervised_learning/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/supervised_learning/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  unsupervised_learning: renderProject({
    tileImage: './portfolio/unsupervised_learning/assets/portfolio_tile.png',
    tileTitle: 'Identifying Customer Segments with Unsupervised Learning',
    markdownUrl: './portfolio/unsupervised_learning/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/unsupervised_learning/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  }),

  word2vec: renderProject({
    tileImage: './portfolio/word2vec/assets/portfolio_tile.png',
    tileTitle: 'Exploring Word Embedding Methods',
    markdownUrl: './portfolio/word2vec/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/word2vec/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
  })
}

// demos

window.demos = {
  cat_dog_classifier: renderDemo({
    demoLink: './portfolio/cat_dog_classifier/index.html',
    demoImage: './portfolio/cat_dog_classifier/assets/demo_tile.png',
    demoDescription: 'Edge Image Classification on Mobile Browsers',
  }),

  docker_tutorial: renderDemo({
    demoLink: 'https://github.com/dylanlrrb/Please-Contain-Yourself',
    demoImage: './portfolio/docker_tutorial/assets/demo_tile.png',
    demoDescription: 'Learn to Use Docker with Hands-on Projects',
  }),

  mnist_nn: renderDemo({
    demoLink: './portfolio/mnist_nn/index.html',
    demoImage: './portfolio/mnist_nn/assets/demo_tile.png',
    demoDescription: 'MNIST Demo: See what a Neural Network is thinkning as it guesses what number you\'re drawing',
  }),
}