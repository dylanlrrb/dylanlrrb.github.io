((demos) => {
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
          .createContextualFragment(demos.map(fragment).join(''))
  

  document.querySelector("#demo-list").append(node)
})([
  {
    demoLink: './portfolio/mnist_nn/index.html',
    demoImage: './portfolio/mnist_nn/assets/demo_tile.png',
    demoDescription: 'MNIST Demo: See what a Neural Network is thinkning as it guesses what number you\'re drawing',
  },
  {
    demoLink: 'https://github.com/dylanlrrb/Please-Contain-Yourself',
    demoImage: './portfolio/docker_tutorial/assets/demo_tile.png',
    demoDescription: 'Learn to Use Docker with Hands-on Projects',
  },
])
