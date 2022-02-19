((demos) => {
  const fragment = ({demoImage, demoDescription, demoLink}) => `
    <a href="${demoLink}" target="_blank" class="demo-tile">
      <div class="demo-image" style="
        background-image:linear-gradient(to right, transparent, transparent, whitesmoke), url(${demoImage});
        background-image:-o-linear-gradient(to right, transparent, transparent, whitesmoke), url(${demoImage});
        background-image:-webkit-gradient(to right, transparent, transparent, whitesmoke), url(${demoImage});">
      </div>
      <div class="demo-description">${demoDescription}</div>
    </a>
  `
  node = document.createRange()
          .createContextualFragment(demos.map(fragment).join(''))
  

  document.querySelector("#demo-list").append(node)
})([
  {
    demoLink: './demos/mnist-demo/index.html',
    demoImage: 'public/images/mnist.png',
    demoDescription: 'MNIST Demo: See what a Neural Network is thinkning as it guesses what number you\'re drawing',
  },
  {
    demoLink: './demos/mnist-demo/index.html',
    demoImage: 'public/images/selfie.jpeg',
    demoDescription: 'this is a demo description',
  },
  {
    demoLink: './demos/mnist-demo/index.html',
    demoImage: 'public/icons/test_icon.png',
    demoDescription: 'this is a demo description',
  },
  {
    demoLink: './demos/mnist-demo/index.html',
    demoImage: 'public/icons/test_icon.png',
    demoDescription: 'this is a demo description',
  },
])
