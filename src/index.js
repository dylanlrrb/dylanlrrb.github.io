projects = ["./src/projects/mnist_nn.js",
            "./src/projects/style_transfer_1.js",
            "./src/projects/backprop_painting.js",
            "./src/projects/supervised_learning.js",
            "./src/projects/docker_tutorial.js",
            "./src/projects/unsupervised_learning.js",
            "./src/projects/signal_separation.js",
            // background removal
            // CFIAR classification final
            // tSNE on penultimate layer of landmark classifier
            // cat/dog classifier in tensorflow (DEMO)
            // De-noising an image and or sound
            // TV script (one to many, just a decoder?)
            // Word to vector (negative sampling explored) 1, 2
            // Sentiment analysis of movie reviews (many to one, just an encoder?) 1
          ]
demos = [
  "./src/demos/mnist_nn.js",
  "./src/demos/docker_tutorial.js",
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

function sequential(p) {
  i = 0
  recurse = () =>  {p[i]().then(() => {i++;if (i >= p.length) {return} else {recurse()}})}
  recurse()
}

window.addEventListener('DOMContentLoaded', () => {
  const hamburger_icon = document.querySelector('header.mobile .hamburger-icon')
  const mobile_nav = document.querySelector('header.mobile .nav')
  const selfie = document.querySelector('div.selfie')

  hamburger_icon.addEventListener('click', () => mobile_nav.classList.toggle('closed'))
  mobile_nav.addEventListener('click', (e) => {
    if (!e.target.classList.contains('nav')) {
      mobile_nav.classList.toggle('closed')
    }
  })
  Array.from(document.querySelectorAll('section')).forEach((section) => {
    section.addEventListener('touchstart', () => mobile_nav.classList.add('closed'))
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
    .then(() => sequential([...projects, ...demos].map(loadScript)))
    .catch(() => console.log('error loading showdown script'))
  
});

// h3 is small centered header under h1