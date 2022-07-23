project_ordered_list = [
  "mnist_nn",
  "backprop_painting",
  "style_transfer_1",
  "docker_tutorial",
  "background_removal",
  "landmark_classification",
  "lstm_scripts",
  "word2vec",
  "cat_dog_classifier",
  "conv_visualizer",
  "super_resolution",
  "image_segmentation",
  "facial_keypoints",
  "yolo",
  // "mask_r_cnn",

  "placeholder",
  "placeholder",
  "placeholder",
  "remove_loader",
]

demo_ordered_list = [
  "mnist_nn",
  "docker_tutorial",
  "cat_dog_classifier",
  "conv_visualizer",
  "super_resolution",
  "image_segmentation",
  "facial_keypoints",
  "yolo",
  // "mask_r_cnn",
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

  years_exp.innerHTML = getYearsSince('9/15/2016')

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

  let filteredTags = []
  const filterButtons =  Array.from(document.querySelectorAll('.project-filters-buttons button'))
  const allFilterButton = document.querySelector('.project-filters-buttons button[data-filter="ALL"]')

  document.querySelector('.project-filters-buttons').addEventListener('click', ({target}) => {
    if (target.tagName !== 'BUTTON') {return}

    if (target.dataset.filter === 'ALL') {
      filterButtons.forEach((b) => b.className = 'dormant-filter-button')
      allFilterButton.className = 'active-filter-button'
      filteredTags = []
      window.project_tiles.forEach((p) => p.classList.remove('display-none'))
    }
    else {
      allFilterButton.className = 'dormant-filter-button'
      if (target.classList.contains('dormant-filter-button')) {
        target.className = 'active-filter-button'
        filteredTags.push(target.dataset.filter)
        window.project_tiles.forEach((p) => {
          p.classList.add('display-none')
          filteredTags.forEach((t) => {
            if (p.dataset.tags.includes(t)) {
              p.classList.remove('display-none')
            }
          })
        })
      } else { // if active already
        target.className = 'dormant-filter-button'
        filteredTags = filteredTags.filter((e) => e !== target.dataset.filter)
        if (filteredTags.length === 0) {
          filterButtons.forEach((b) => b.className = 'dormant-filter-button')
          allFilterButton.className = 'active-filter-button'
          filteredTags = []
          window.project_tiles.forEach((p) => p.classList.remove('display-none'))
        } else {
          window.project_tiles.forEach((p) => {
            p.classList.add('display-none')
            filteredTags.forEach((t) => {
              if (p.dataset.tags.includes(t)) {
                p.classList.remove('display-none')
              }
            })
          })
        }
      }
    }
  })
  
  
});
