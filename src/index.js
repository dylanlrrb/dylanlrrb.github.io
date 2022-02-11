window.addEventListener('DOMContentLoaded', (event) => {
  hamburger_icon = document.querySelector('header.mobile .hamburger-icon')
  x_icon = document.querySelector('header.mobile .x-icon')
  mobile_nav = document.querySelector('header.mobile .nav')
  selfie = document.querySelector('div.selfie')

  hamburger_icon.addEventListener('click', () => mobile_nav.classList.toggle('closed'))
  mobile_nav.addEventListener('click', (e) => {
    if (!e.target.classList.contains('nav')) {
      mobile_nav.classList.toggle('closed')
    }
  })

  node = document.createElement('div')
  node.classList.add('transparent')
  node.classList.add('high')
  setTimeout(() => {
    selfie.appendChild(node)
  }, 200)
  setTimeout(() => node.classList.toggle('transparent'), 300)
  
});