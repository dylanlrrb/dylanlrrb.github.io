projects = ["./src/projects/mnist_demo.js",
            "./src/projects/backprop_painting.js",
            "./src/projects/supervised_learning.js",
            "./src/projects/supervised_learning.js",
            "./src/projects/backprop_painting.js",
            "./src/projects/mnist_demo.js",]
            

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
    .then(() => {
      sequential(projects.map(loadScript))
  })
  .catch(() => console.log('error loading showdown script'))

  loadScript('./src/projects/00_demos.js')()
  
});

// h3 is small centered header under h1