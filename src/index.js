project_ordered_list = [
  "mnist_nn",
  "backprop_painting",
  "style_transfer_1",
  "docker_tutorial",
  // "supervised_learning",
  // "unsupervised_learning",
  // "signal_separation",
  "background_removal",
  "cifar10_classification",
  "landmark_classification",
  "lstm_scripts",
  "movie_sentiment",
  "denoise_autoencoder",
  "word2vec",
  "cat_dog_classifier",
  // GANs + deployment
  // ---
  // View feature maps in a real time web app 
  // super resolution notebook and webapp
  // Multi box detector using mobile net + demo app (first expor`ation in pytorch then implementationin a web app)
  // Class activation Map/ Saliency Map/ Attribution Maps
  // Deep dream clone
  // ---
  // style transfer pt 2, model per style to transfer 1
  // style transfter pt 3, arbitrary style transfer in browser (DEMO)
  // project with seq2seq without attention (translation in order to compare to w/ attention?)
  // Translation with Attention project (seq2seq w/ attention) + class activation map of attention matrix as sentence is translated 1, 2, 3, 4
  // Image captioning with attention, visualizing focused parts of image attention is given to 1
  // brainwave signal separation and deep learning 
  // ---
  // Gan projects
  // ---
  // Renforcemnt learning course
  // ---
  // dermatologist AI
  // Create images from text
  // Add color to a BW image
  // customer segmentation https://www.kaggle.com/fabiendaniel/customer-segmentation
  // genomics PCA
  "placeholder",
  "placeholder",
  "placeholder",
  "remove_loader",
]

demo_ordered_list = [
  "mnist_nn",
  "docker_tutorial",
  "cat_dog_classifier",
  // View feature maps in a real time web app
  // Multi box detector using mobile net + demo app (first exporation in pytorch then implementationin a web app)
  // style transfter pt 3, arbitrary style transfer in browser (DEMO) 
]


            

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

function getYearsSince(start) {
  const startDate = new Date(start);
  // One year in milliseconds
  const oneYear = 1000 * 60 * 60 * 24 * 365;
  // Calculating the time difference between two dates
  const diffInTime = Date.now() - startDate.getTime();
  // Calculating the no. of days between two dates
  const diffInYears = Math.round(diffInTime / oneYear);
  return diffInYears;
}

window.addEventListener('DOMContentLoaded', () => {
  const hamburger_icon = document.querySelector('header.mobile .hamburger-icon')
  const mobile_nav = document.querySelector('header.mobile .nav')
  const selfie = document.querySelector('div.selfie')
  const years_exp = document.querySelector('.years-exp')

  years_exp.innerHTML = getYearsSince('9/15/2016', )

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
    setTimeout(() => node.classList.toggle('transparent'), 500)
  }, 100)
  

  loadScript('https://cdnjs.cloudflare.com/ajax/libs/showdown/2.0.0/showdown.min.js')()
    .then(() => loadScript('./src/render_tiles.js')())
    .then(() => {
      project_ordered_list.map((p) => window.projects[p]())
      demo_ordered_list.map((d) => window.demos[d]())
    })
    .catch((e) => console.log('error loading showdown script', e))
  
});
