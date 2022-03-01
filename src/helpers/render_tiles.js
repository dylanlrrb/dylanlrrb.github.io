const renderProject = ({tileImage, tileTitle, markdownUrl, controls}) => {

  if (!tileImage) {
    const node = document.createRange().createContextualFragment(`<div class="tile-placeholder"></div>`);
    document.querySelector("#projects").append(node)
    return
  }

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

const renderDemo = (demo) => {
  const fragment = ({demoImage, demoDescription, demoLink}) => `
    <a href="${demoLink}" target="_blank" class="demo-tile">
      <div class="demo-image" style="
        background-image:linear-gradient(to right, transparent, transparent, white), url(${demoImage});
        background-image:-o-linear-gradient(to right, transparent, transparent, white), url(${demoImage});
        background-image:-webkit-gradient(to right, transparent, transparent, white), url(${demoImage});">
      </div>
      <div class="demo-description">${demoDescription}</div>
    </a>
  `
  node = document.createRange()
          .createContextualFragment(fragment(demo))
  

  document.querySelector("#demo-list").append(node)
}