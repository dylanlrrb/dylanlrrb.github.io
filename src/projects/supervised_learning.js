
(({tileImage, tileTitle, markdownUrl, controls}) => {
  
  const html = `<div class="project">
    <div class="tile">
      <div class="tile-image" style="background-image:url(${tileImage})"></div>
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
  const modal = node.querySelector('.modal')
  const exit = modal.querySelector('.exit')
  const scrim = modal.querySelector('.modal-scrim')
  const inner = modal.querySelector('.modal-inner')
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

  document.querySelector("#projects").append(node)

  const converter = new showdown.Converter()
  fetch(markdownUrl).then(res => res.text())
    .then(text => {
      inner.innerHTML = converter.makeHtml(text)
      document.querySelector("#projects").append(node)
    })
})({
  tileImage: './public/images/octo_style.gif',
  tileTitle: 'Supervised Learning Comparison',
  markdownUrl: 'https://raw.githubusercontent.com/dylanlrrb/Please-Contain-Yourself/master/README.md',
  controls: [
    '<a href=" https://colab.research.google.com/github/dylanlrrb/portfolio/blob/main/backprop_painting.ipynb" target="_blank"><button class="secondary-action-button">Notebook</button></a>',
    '<a href="" target="_blank"><button class="primary-action-button">Demo</button></a>'
  ],
})
