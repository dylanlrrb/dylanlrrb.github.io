projects = ["./src/projects/mnist-demo.html",
            "./src/projects/supervised_learning.html",
            "./src/projects/supervised_learning.html",
            "./src/projects/supervised_learning.html",
            "./src/projects/supervised_learning.html",
            "./src/projects/supervised_learning.html"]

window.addEventListener('DOMContentLoaded', (event) => {
  const hamburger_icon = document.querySelector('header.mobile .hamburger-icon')
  const mobile_nav = document.querySelector('header.mobile .nav')
  const selfie = document.querySelector('div.selfie')

  hamburger_icon.addEventListener('click', () => mobile_nav.classList.toggle('closed'))
  mobile_nav.addEventListener('click', (e) => {
    if (!e.target.classList.contains('nav')) {
      mobile_nav.classList.toggle('closed')
    }
  })

  const node = document.createElement('div')
  node.classList.add('transparent')
  node.classList.add('high')
  setTimeout(() => {
    selfie.appendChild(node)
  }, 100)
  setTimeout(() => node.classList.toggle('transparent'), 300)

  Promise.all(projects.map(p => fetch(p).then(r => r.text())))
    .then(data => {
      data.forEach(d =>  document.querySelector("#projects").innerHTML += d)
    })
    .then(() => {
      document.querySelectorAll('#projects .project')
        .forEach(project => {
          const tile = project.querySelector('.tile')
          const modal = project.querySelector('.modal')
          const exit = modal.querySelector('.exit')
          const scrim = modal.querySelector('.modal-scrim')
          tile.addEventListener('click', () => {
            modal.classList.toggle('display-none')
            document.documentElement.classList.toggle('disable-scroll')
          })
          exit.addEventListener('click', () => {
            modal.classList.toggle('display-none')
            document.documentElement.classList.toggle('disable-scroll')
          })
          scrim.addEventListener('click', () => {
            modal.classList.toggle('display-none')
            document.documentElement.classList.toggle('disable-scroll')
          })
        })
    })

  
});