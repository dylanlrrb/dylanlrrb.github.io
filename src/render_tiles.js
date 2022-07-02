window.project_tiles = []

const renderProject = ({projectId, tileImage, tileTitle, markdownUrl, controls, tags=[]}) => {
  const html = `<div class="project" data-tags="${tags.join(',')}">
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
          <img class="share-link" src="public/icons/share-icon.png" alt="Share link" />
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
  const share = modal.querySelector('.modal-controls .share-link')

  share.addEventListener('click', () => {
    navigator.clipboard.writeText(`${window.location.origin}/?project=${projectId}#portfolio`)
      .then(() => {
        alert("Copied link to clipboard");
      })
  })
  
  return () => {
    document.querySelector("#projects").append(node)
    window.project_tiles.push(tile.parentElement)
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
    inner.innerHTML = `<div class="lds-ring"><div class="blue"></div><div></div><div></div><div></div></div>`
    const converter = new showdown.Converter({strikethrough:true, tables:true, tasklists:true, emoji:true, openLinksInNewWindow:true})
    const queryParam = (new URLSearchParams(window.location.search)).get('project')
    if (projectId === queryParam) {
      fetch(markdownUrl).then(res => res.text())
      .then(text => {
        inner.innerHTML = converter.makeHtml(text)
        document.querySelector("#projects").append(node)
      })
      const click = new Event('click')
      tile.dispatchEvent(click)
    }
    setTimeout(() => {
      tileIm.style['background-image'] = `url(${tileImage})`
      if (projectId !== queryParam) {
        fetch(markdownUrl).then(res => res.text())
          .then(text => {
            inner.innerHTML = converter.makeHtml(text)
            document.querySelector("#projects").append(node)
        })
      }
    }, 1500)
  }
}

const renderDemo = (demo) => {
  const {demoImage, demoDescription, demoLink} = demo
  const fragment = `<a href="${demoLink}" class="demo-tile">
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

const TAGS = {
  BIOINFORMATICS: "BIOINFORMATICS",
  COMPUTER_VISION: "COMPUTER_VISION",
  DEEP_LEARNING: "DEEP_LEARNING",
  DEPLOYMENT: "DEPLOYMENT",
  RECURRENCE: "RECURRENCE",
  SIGNAL_PROCESSING: "SIGNAL_PROCESSING",
  NATURAL_LANGUAGE_PROCESSING: "NATURAL_LANGUAGE_PROCESSING",
  
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
    projectId: 'background_removal',
    tileImage: './portfolio/background_removal/assets/portfolio_tile.gif',
    tileTitle: 'Using Gaussian Mixture Models to Isolate Movement in Video',
    markdownUrl: './portfolio/background_removal/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/background_removal/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEEP_LEARNING],
  }),

  backprop_painting: renderProject({
    projectId: 'backprop_painting',
    tileImage: './portfolio/backprop_painting/assets/portfolio_tile.png',
    tileTitle: 'Visualizing Convolutional Layers in a Trained VGG Network',
    markdownUrl: './portfolio/backprop_painting/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/backprop_painting/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEEP_LEARNING],
  }),

  cat_dog_classifier: renderProject({
    projectId: 'cat_dog_classifier',
    tileImage: './portfolio/cat_dog_classifier/assets/portfolio_tile.png',
    tileTitle: 'Classifying Cat Vs. Dog Breeds with a MobileNet',
    markdownUrl: './portfolio/cat_dog_classifier/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/cat_dog_classifier/notebook.ipynb" target="_blank">
        <button class="secondary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/cat_dog_classifier/demo/build/index.html">
        <button class="primary-action-button">Demo</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEPLOYMENT, TAGS.DEEP_LEARNING],
  }),

  cifar10_classification: renderProject({
    projectId: 'cifar10_classification',
    tileImage: './portfolio/cifar10_classification/assets/portfolio_tile.jpeg',
    tileTitle: 'Exploring Effect of Image Resizing on Classification of CIFAR-10 dataset',
    markdownUrl: './portfolio/cifar10_classification/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/cifar10_classification/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEEP_LEARNING],
  }),

  conv_visualizer: renderProject({
    projectId: 'conv_visualizer',
    tileImage: './portfolio/conv_visualizer/assets/portfolio_tile.png',
    tileTitle: 'Visualizing Convolutional Layers of a Trained CNN',
    markdownUrl: './portfolio/conv_visualizer/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/conv_visualizer/notebook.ipynb" target="_blank">
        <button class="secondary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/conv_visualizer/demo/build/index.html">
        <button class="primary-action-button">Demo</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEPLOYMENT, TAGS.DEEP_LEARNING],
  }),

  docker_tutorial: renderProject({
    projectId: 'docker_tutorial',
    tileImage: './portfolio/docker_tutorial/assets/portfolio_tile.png',
    tileTitle: 'Project-Based Docker Tutorial',
    markdownUrl: 'https://raw.githubusercontent.com/dylanlrrb/Please-Contain-Yourself/master/README.md',
    controls: [
      `<a href="https://github.com/dylanlrrb/Please-Contain-Yourself" target="_blank">
        <button class="primary-action-button">Tutorial</button></a>`,
    ],
  }),

  landmark_classification: renderProject({
    projectId: 'landmark_classification',
    tileImage: './portfolio/landmark_classification/assets/portfolio_tile.png',
    tileTitle: 'Implementing Reverse Image Search with Landmark Images',
    markdownUrl: './portfolio/landmark_classification/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/landmark_classification/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEEP_LEARNING],
  }),

  lstm_scripts: renderProject({
    projectId: 'lstm_scripts',
    tileImage: './portfolio/lstm_scripts/assets/portfolio_tile.gif',
    tileTitle: 'Generating  Parks and Rec Episode Scripts with LSTM Recurrent Neural Networks',
    markdownUrl: './portfolio/lstm_scripts/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/lstm_scripts/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.DEEP_LEARNING, TAGS.RECURRENCE],
  }),

  mnist_nn: renderProject({
    projectId: 'mnist_nn',
    tileImage: './portfolio/mnist_nn/assets/portfolio_tile.gif',
    tileTitle: 'Real Time Activation and Weight Visualization of Neural Network Trained on MNIST Dataset',
    markdownUrl: './portfolio/mnist_nn/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/neural-networks-from-scratch/blob/main/network2.ipynb" target="_blank">
        <button class="secondary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/mnist_nn/demo/index.html" target="_blank"><button class="primary-action-button">Demo</button></a>`
    ],
    tags: [TAGS.DEEP_LEARNING],
  }),

  movie_sentiment: renderProject({
    projectId: 'movie_sentiment',
    tileImage: './portfolio/movie_sentiment/assets/portfolio_tile.png',
    tileTitle: 'Sentiment Analysis of Movie Reviews with LSTM Recurrent Neural Nets',
    markdownUrl: './portfolio/movie_sentiment/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/movie_sentiment/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.DEEP_LEARNING, TAGS.RECURRENCE],
  }),

  image_segmentation: renderProject({
    projectId: 'image_segmentation',
    tileImage: './portfolio/image_segmentation/assets/portfolio_tile.png',
    tileTitle: 'Image Segmentation with U-Nets and COCO Dataset',
    markdownUrl: './portfolio/image_segmentation/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/image_segmentation/notebook.ipynb" target="_blank">
        <button class="secondary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/image_segmentation/demo/build/index.html">
        <button class="primary-action-button">Demo</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEPLOYMENT, TAGS.DEEP_LEARNING],
  }),

  style_transfer_1: renderProject({
    projectId: 'style_transfer_1',
    tileImage: './portfolio/style_transfer_1/assets/portfolio_tile.gif',
    tileTitle: 'Style Transfer with Backpropogation Through a Convolutional Neural Network',
    markdownUrl: './portfolio/style_transfer_1/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/style_transfer_1/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEEP_LEARNING],
  }),

  super_resolution: renderProject({
    projectId: 'super_resolution',
    tileImage: './portfolio/super_resolution/assets/portfolio_tile.png',
    tileTitle: 'Super Resolving Images with GANs',
    markdownUrl: './portfolio/super_resolution/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/super_resolution/notebook.ipynb" target="_blank">
        <button class="secondary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/super_resolution/demo/build/index.html">
        <button class="primary-action-button">Demo</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEPLOYMENT, TAGS.DEEP_LEARNING],
  }),

  word2vec: renderProject({
    projectId: 'word2vec',
    tileImage: './portfolio/word2vec/assets/portfolio_tile.png',
    tileTitle: 'Exploring Word Embedding Methods',
    markdownUrl: './portfolio/word2vec/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/word2vec/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
    ],
    tags: [TAGS.NATURAL_LANGUAGE_PROCESSING, TAGS.DEEP_LEARNING],
  }),

  yolo: renderProject({
    projectId: 'yolo',
    tileImage: './portfolio/yolo/assets/portfolio_tile.png',
    tileTitle: 'YOLO Object Detection',
    markdownUrl: './portfolio/yolo/README.md',
    controls: [
      `<a href="https://colab.research.google.com/github/dylanlrrb/dylanlrrb.github.io/blob/master/portfolio/yolo/notebook.ipynb" target="_blank">
        <button class="primary-action-button">Notebook</button></a>`,
      `<a href="./portfolio/yolo/demo/build/index.html">
        <button class="primary-action-button">Demo</button></a>`,
    ],
    tags: [TAGS.COMPUTER_VISION, TAGS.DEPLOYMENT, TAGS.DEEP_LEARNING],
  }),
}

// demos

window.demos = {
  cat_dog_classifier: renderDemo({
    demoLink: './portfolio/cat_dog_classifier/demo/build/index.html',
    demoImage: './portfolio/cat_dog_classifier/assets/demo_tile.png',
    demoDescription: 'Edge Pet and Breed Classification on Mobile Browsers',
  }),

  conv_visualizer: renderDemo({
    demoLink: './portfolio/conv_visualizer/demo/build/index.html',
    demoImage: './portfolio/conv_visualizer/assets/demo_tile.png',
    demoDescription: 'Visualize Convolutional Layers as it Extracts Features from an Image',
  }),

  docker_tutorial: renderDemo({
    demoLink: 'https://github.com/dylanlrrb/Please-Contain-Yourself',
    demoImage: './portfolio/docker_tutorial/assets/demo_tile.png',
    demoDescription: 'Learn to Use Docker with Hands-on Projects',
  }),

  mnist_nn: renderDemo({
    demoLink: './portfolio/mnist_nn/demo/index.html',
    demoImage: './portfolio/mnist_nn/assets/demo_tile.png',
    demoDescription: 'MNIST Demo: See what a Neural Network is thinkning as it guesses what number you\'re drawing',
  }),

  mask_r_cnn: renderDemo({
    demoLink: "./portfolio/mask_r_cnn/demo/build/index.html",
    demoImage: './portfolio/mask_r_cnn/assets/demo_tile.png',
    demoDescription: 'Object Detection with Segmentation',
  }),

  super_resolution: renderDemo({
    demoLink: './portfolio/super_resolution/demo/build/index.html',
    demoImage: './portfolio/super_resolution/assets/demo_tile.png',
    demoDescription: 'Super Resolving Images with a Generative Adversarial Network',
  }),

  image_segmentation: renderDemo({
    demoLink: './portfolio/image_segmentation/demo/build/index.html',
    demoImage: './portfolio/image_segmentation/assets/demo_tile.png',
    demoDescription: 'Image Segmentation Common Object in Contect (COCO) with U-Nets',
  }),

  yolo: renderDemo({
    demoLink: "./portfolio/yolo/demo/build/index.html",
    demoImage: './portfolio/yolo/assets/demo_tile.png',
    demoDescription: 'YOLO Object Detection',
  }),
}