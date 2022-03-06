// model data
let data
let models
let selectedHiddenNode
let clearedInputArea = true

// Elements
let inputArea
let neuronActivations
let canvas

// Model
let canDraw = true
const inputLayerModel = new Array(28).fill(null).map(() => new Array(28).fill(0))
const model_props = {
  "cost-function": "xcost",
  "weight-decay": '2l',
  "early-stopping": '1stop',
}

// View Elements
const inputLayerElements = new Array(28).fill(null).map(() => new Array(28).fill(null).map(() => resetNodeColorAndOpacity(createNode({tag: 'div', classNames: ['input-node']}))))
inputLayerElements.forEach((row, i) => {
  row.forEach((el, j) => {
    el.dataset.row = i
    el.dataset.col = j
  })
})
const firstLayerElements = new Array(25).fill(null).map(() => createNode({tag: 'div', classNames: ['hidden-layer-node', 'pointer']}))
const secondLayerElements = new Array(25).fill(null).map(() => createNode({tag: 'div', classNames: ['hidden-layer-node']}))
const thirdLayerElements = new Array(10).fill(null).map(() => createNode({tag: 'div', classNames: ['output-layer-node']}))
const layerElements = [firstLayerElements, secondLayerElements, thirdLayerElements]

// Helpers
function createNode ({ tag, classNames=[] }) {
  const el = document.createElement(tag)
  classNames.forEach((name) => {
    el.classList.add(name)
  })
  return el
}

function activationToHiddenNodeGradient (activation, node) {
  node.style.backgroundImage = `linear-gradient(transparent 0 ${Math.floor((1-activation)*100)}%, black ${Math.floor((1-activation)*100)}% 100%)`
  return node
}

function activationToNodeOpacity (activation, node) {
  node.style.opacity = `${activation}`
  return node
}

function weightToNodeColor (weight, node) {
  if (weight > 0) {
    node.style.backgroundColor = '#0000ff'
    node.style.borderColor = '#0000ff'
  } else {
    node.style.backgroundColor = '#ff0000'
    node.style.borderColor = '#ff0000'
  }
  node.style.opacity = `${Math.abs(sigmoid(weight))}`
  return node
}

function overlayDrawnWeights (weight, node) {
  const norm_weight = 1-weight
  let rgb = node.style.backgroundColor
  let [r, g, b] = rgb.slice(
      rgb.indexOf("(") + 1, 
      rgb.indexOf(")")
  ).split(", ")
  const new_color = `#${Math.floor(norm_weight*r).toString(16).padStart(2, '0')}${Math.floor(norm_weight*g).toString(16).padStart(2, '0')}${Math.floor(norm_weight*b).toString(16).padStart(2, '0')}`
  node.style.backgroundColor = new_color
}

function resetNodeColorAndOpacity (node) {
  node.style.backgroundColor = 'black'
  node.style.borderColor = 'black'
  node.style.opacity = '0.1'
  return node
}

function sigmoid(t) {
  return 1/(1+Math.pow(Math.E, -t));
}

function zip(a, b) {
  return a.map(function(e, i) {
    return [e, b[i]];
  })
}

const mmultiply = (a, b) => a.map(x => transpose(b).map(y => dotproduct(x, y)));
const dotproduct = (a, b) => a.map((x, i) => a[i] * b[i]).reduce((m, n) => m + n);
const transpose = a => a[0].map((x, i) => a.map(y => y[i]));

// Handlers
function showInputLayerWeights(neuron) {
  canDraw = false
  data.weights[0][neuron].forEach((weight, i) => {
    weightToNodeColor(weight, inputLayerElements[Math.floor(i/28)][i%28])
    overlayDrawnWeights(inputLayerModel[Math.floor(i/28)][i%28], inputLayerElements[Math.floor(i/28)][i%28])
  })
}

function resetInputLayer() {
  inputLayerElements.forEach((row, i) => {
    row.forEach((el, j) => {
      inputLayerModel[i][j] = 0
      resetNodeColorAndOpacity(el)
    })
  })
  canDraw = true
}

function resetNeuronGradients() {
  layerElements.forEach((layer) => {
    layer.forEach((el) => activationToHiddenNodeGradient(0, el))
  })
}

function renderInputArea() {
  inputArea = document.querySelector("#input-layer")
  neuronActivations = document.querySelector("#neuron-activations")

  const inputAreaFragment = new DocumentFragment()
  inputLayerElements.forEach((row) => {
    const r = createNode({tag: 'div', classNames: ['input-row']})
    row.forEach((el) => {
      r.appendChild(el)
    })
    inputAreaFragment.appendChild(r)
  })
  inputArea.appendChild(inputAreaFragment)
}

function renderLayers() {
  neuronActivations = document.querySelector("#neuron-activations")
  const neuronActivationsFragment = new DocumentFragment()
  const layerOne = createNode({tag: 'div', classNames: ['activation-row']})
  firstLayerElements.forEach((el) => layerOne.appendChild(el))
  const layerTwo = createNode({tag: 'div', classNames: ['activation-row']})
  secondLayerElements.forEach((el) => layerTwo.appendChild(el))
  const layerThree = createNode({tag: 'div', classNames: ['activation-row']})
  thirdLayerElements.forEach((el) => layerThree.append(el))
  neuronActivationsFragment.appendChild(layerThree)
  neuronActivationsFragment.appendChild(layerTwo)
  neuronActivationsFragment.appendChild(layerOne)
  neuronActivations.appendChild(neuronActivationsFragment)
}

function renderLayerWeights() {
  canvas = document.querySelector('#neuron-weights')
  const ctx = canvas.getContext("2d")
  ctx.clearRect(0, 0, 600, 400);

  data.weights[2].forEach((weights, input) => {
    weights.forEach((weight, output) => {
      ctx.beginPath();
      if (weight > 0) {
        ctx.strokeStyle = `#0000ff${Math.floor((Math.abs(sigmoid(weight))*255)).toString(16)}`
      } else {
        ctx.strokeStyle = `#ff0000${Math.floor((Math.abs(sigmoid(weight))*255)).toString(16)}`
      }
      ctx.lineWidth = 0.5
      ctx.moveTo(60*input+27, 54)
      ctx.lineTo(24*output+10, 149)
      ctx.stroke();
    })
  })

  data.weights[1].forEach((weights, input) => {
    weights.forEach((weight, output) => {
      ctx.beginPath();
      if (weight > 0) {
        ctx.strokeStyle = `#0000ff${Math.floor((Math.abs(sigmoid(weight))*255)).toString(16)}`
      } else {
        ctx.strokeStyle = `#ff0000${Math.floor((Math.abs(sigmoid(weight))*255)).toString(16)}`
      }
      ctx.lineWidth = 0.25
      ctx.moveTo(24*input+10, 171)
      ctx.lineTo(24*output+10, 267)
      ctx.stroke();
    })
  })
}

function updateModel(modelName) {
  data = models[modelName]
  resetNeuronGradients()
  renderLayerWeights()
  if (!clearedInputArea) {
    calculateAndUpdateActivations(inputLayerModel)
  } 
  if (!canDraw) {
    showInputLayerWeights(selectedHiddenNode)
  }
}

function draw(e) {
  // console.log(e)
  const eventType = e instanceof TouchEvent ? true : e.buttons > 0
  // if (canDraw && e.buttons > 0 && e.target.classList.contains('input-node')) {
  if (canDraw && eventType && e.target.classList.contains('input-node')) {
    clearedInputArea = false
    const target = e instanceof TouchEvent ? document.elementFromPoint(e.changedTouches[0].clientX, e.changedTouches[0].clientY) : e.target
    const row = parseInt(target.dataset.row, 10)
    const col = parseInt(target.dataset.col, 10)
    
    if (row < 4 || row > 23 || col < 4 || col > 23) {
      return
    }

    const centerNode = inputLayerElements[row][col]
    const upNode = (inputLayerElements[row-1] || [])[col]
    const downNode = (inputLayerElements[row+1] || [])[col]
    const leftNode = inputLayerElements[row][col-1]
    const rightNode = inputLayerElements[row][col+1]
    const upperLeftNode = (inputLayerElements[row-1] || [])[col-1]
    const lowerRightNode = (inputLayerElements[row+1] || [])[col+1]
    const lowerLeftNode = inputLayerElements[row+1][col-1]
    const upperRightNode = inputLayerElements[row-1][col+1]


    centerNode.style.opacity = `${Math.min(parseFloat(centerNode.style.opacity) + 0.6, 1)}`
    inputLayerModel[row][col] = Math.min(inputLayerModel[row][col] + 0.6, 1)
    if (upNode) {
      upNode.style.opacity = `${Math.min(parseFloat(upNode.style.opacity) + 0.2, 1)}`
      inputLayerModel[row-1][col] = Math.min(inputLayerModel[row-1][col] + 0.2, 1)
    }
    if (downNode) {
      downNode.style.opacity = `${Math.min(parseFloat(downNode.style.opacity) + 0.2, 1)}`
      inputLayerModel[row+1][col] = Math.min(inputLayerModel[row+1][col] + 0.2, 1)
    }
    if (leftNode) {
      leftNode.style.opacity = `${Math.min(parseFloat(leftNode.style.opacity) + 0.2, 1)}`
      inputLayerModel[row][col-1] = Math.min(inputLayerModel[row][col-1] + 0.2, 1)
    }
    if (rightNode) {
      rightNode.style.opacity = `${Math.min(parseFloat(rightNode.style.opacity) + 0.2, 1)}`
      inputLayerModel[row][col+1] = Math.min(inputLayerModel[row][col+1] + 0.2, 1)
    }
    if (upperLeftNode) {
      upperLeftNode.style.opacity = `${Math.min(parseFloat(upperLeftNode.style.opacity) + 0.1, 1)}`
      inputLayerModel[row-1][col-1] = Math.min(inputLayerModel[row-1][col-1] + 0.1, 1)
    }
    if (lowerRightNode) {
      lowerRightNode.style.opacity = `${Math.min(parseFloat(lowerRightNode.style.opacity) + 0.1, 1)}`
      inputLayerModel[row+1][col+1] = Math.min(inputLayerModel[row+1][col+1] + 0.1, 1)
    }
    if (lowerLeftNode) {
      lowerLeftNode.style.opacity = `${Math.min(parseFloat(lowerLeftNode.style.opacity) + 0.1, 1)}`
      inputLayerModel[row+1][col-1] = Math.min(inputLayerModel[row+1][col-1] + 0.1, 1)
    }
    if (upperRightNode) {
      upperRightNode.style.opacity = `${Math.min(parseFloat(upperRightNode.style.opacity) + 0.1, 1)}`
      inputLayerModel[row-1][col+1] = Math.min(inputLayerModel[row-1][col+1] + 0.1, 1)
    }
    calculateAndUpdateActivations(inputLayerModel)
  }
}

function calculateAndUpdateActivations(inputActivation) {
  let a = inputActivation.flat().map(x => [x])
  zip(data.biases, data.weights).forEach(([b, w], i) => {
    a = mmultiply(w, a).map((el, k) => el[0]+b[k][0]).map((el) => [sigmoid(el)])
    layerElements[i].forEach((el, j) => activationToHiddenNodeGradient(a[j][0], el))
  })
  return a
}

function throttle (func, limit) {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
}

// Setup Event Listeners
firstLayerElements.forEach((el, i) => {
  el.addEventListener('click', () => {
    firstLayerElements.forEach((node, i) => {
      node.classList.remove('highlighted')
    })
    el.classList.add('highlighted')
    inputLayerModel.forEach((row, j) => {
      row.forEach((activation, i) => {
        inputLayerElements[j][i].style.opacity = activation
      })
    })
    selectedHiddenNode = i
    showInputLayerWeights(selectedHiddenNode)
  })
})

document.addEventListener("DOMContentLoaded", () => {
  models = {
    "data_qcost_0l_0stop_": data_qcost_0l_0stop,
    "data_qcost_1l_0stop_": data_qcost_1l_0stop,
    "data_qcost_2l_0stop_": data_qcost_2l_0stop,
    "data_qcost_0l_1stop_": data_qcost_0l_1stop,
    "data_qcost_1l_1stop_": data_qcost_1l_1stop,
    "data_qcost_2l_1stop_": data_qcost_2l_1stop,
    "data_xcost_0l_0stop_": data_xcost_0l_0stop,
    "data_xcost_1l_0stop_": data_xcost_1l_0stop,
    "data_xcost_2l_0stop_": data_xcost_2l_0stop,
    "data_xcost_0l_1stop_": data_xcost_0l_1stop,
    "data_xcost_1l_1stop_": data_xcost_1l_1stop,
    "data_xcost_2l_1stop_": data_xcost_2l_1stop,
  }
  
  data = models["data_xcost_2l_1stop_"]

  renderInputArea()
  renderLayers()
  renderLayerWeights()

  document.querySelector('#clear-button').addEventListener('click', () => {
    clearedInputArea = true
    resetNeuronGradients()
    resetInputLayer()
    firstLayerElements.forEach((el) => {
      el.classList.remove('highlighted')
    })
  })
  inputArea.addEventListener('mousemove', throttle(draw, 5))
  inputArea.addEventListener('touchmove', throttle(draw, 5))
  inputArea.addEventListener('touchstart', () => {
    document.querySelector('body').classList.add('disable-scroll')
  })
  inputArea.addEventListener('touchend', () => {
    document.querySelector('body').classList.remove('disable-scroll')
  })
  document.querySelector('#cost_function_selector').addEventListener('change', (e) => {
    const {name, value} = e.target
    model_props[name] = value
    let var_name = "data_"
    for (const key in model_props) {
      var_name += `${model_props[key]}_`
    }
    updateModel(var_name)
  })
  document.querySelector('#regularization_selector').addEventListener('change', (e) => {
    const {name, value} = e.target
    model_props[name] = value
    let var_name = "data_"
    for (const key in model_props) {
      var_name += `${model_props[key]}_`
    }
    updateModel(var_name)
  })
})