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
      
    })
  })({
    tileImage: './public/images/mnist-demo2.gif',
    tileTitle: 'Real Time Activation and Weight Visualization of Neural Network Trained on MNIST Datase',
    markdownUrl: 'https://raw.githubusercontent.com/dylanlrrb/dylanlrrb.github.io/master/demos/mnist-demo/README.md',
    controls: [
      '<a href="https://colab.research.google.com/github/dylanlrrb/neural-networks-from-scratch/blob/main/network2.ipynb" target="_blank"><button class="secondary-action-button">Notebook</button></a>',
      '<a href="./demos/mnist-demo/index.html" target="_blank"><button class="primary-action-button">Demo</button></a>'
    ],
  })
