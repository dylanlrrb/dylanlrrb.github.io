import React from 'react';
import * as tf from "@tensorflow/tfjs"
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'
import './App.css';
import Camera from './components/Camera/Camera'
import Info from './components/Info/Info'
import Results from './components/Results/Results';
import Debug from './components/Debug/Debug';

const pixel_count_thresh = 1000
const max_classes = 5

const idx_to_category_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'background']

function idx_to_color(idx) {
  const bg = [0,0,0,0]
  const colors = [
    [255,0,0,255],   //red
    [255,128,0,255], //orange
    [255,255,0,255], //yellow
    [0,255,0,255],   //green
    [0,255,255,255], //cyan
    [0,0,255,255],   //blue
    [127,0,255,255], //purple
    [255,0,255,255], //magenta
  ]
  if (idx === 80) {
    return bg
  }
  return colors[idx % 8]
}

function colorString(values) {
  let [r,g,b,a] = values
  return `rgba(${r},${g},${b},${a/500})`
}

class App extends React.Component {
  constructor(props) {
    super(props);

    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/finetune_mobilnet_u_sep_1_tvl-gamma/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/mobilnet_u_sep_1_tvl-gamma/model.json'
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/properly_preprocessed/model.json' // best so far
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/u_sep_1_fine/model.json' 
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/u_sep_1/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/u_sep_2/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/u_net_1/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/u_jacard_1/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/image_segmentation/u_jacard_2/model.json'

    this.state = {
      model: undefined,
      loading: true,
      masks: undefined,
      softmax_threshold: 0.99,
      classes: [],
      logs: [],
      paused: false,
    }

    this.debug = {
      log: ((message) => {
        const {logs} = this.state
        !this.state.paused && logs.push({type: 'log', message})
        this.state.logs.length > 100 && logs.shift()
        this.setState({logs})
      }),
      error: ((message) => {
        const {logs} = this.state
        !this.state.paused && logs.push({type: 'error', message})
        this.state.logs.length > 100 && logs.shift()
        this.setState({logs})
      }),
      pause: (() => {this.setState({paused: true})}),
      resume: (() => {this.setState({paused: false})}),
      clear: (() => {this.setState({logs: []})}),
    }
  }

  componentDidCatch(error, errorInfo) {
    this.debug.error(error, errorInfo);
  }

  componentDidMount = async () => {
    const model = await tf.loadGraphModel(this.modelURL);
    this.setState({
      loading: false,
      model
    })
  }

  updateSoftmaxThresh = (newThresh) => {
    this.setState({softmax_threshold: newThresh})
  }

  predict = async (tensor) => {
    if (this.state.model) {
      let [masks, classes] = tf.tidy(() => {
          tensor = tf.image.resizeNearestNeighbor(tensor, [224,224]).toFloat()
          tensor = tf.expandDims(tensor, 0)
          const predictions = tf.squeeze(this.state.model.predict(tensor))
          const sparse_classes = tf.argMax(predictions, 2).dataSync()
          const class_probs = tf.max(predictions, 2).dataSync()

          let classes = sparse_classes.reduce((acc, filter_idx, i) => {
            if (class_probs[i] > this.state.softmax_threshold) {
              acc[filter_idx] += 1
            }
            return acc
          }, (new Array(81)).fill(0))

          classes = classes
                      .map((count, idx) => ({count, idx}))
                      .filter((x) => x.idx !== 80)
                      .sort((a, b) => b.count - a.count)
                      .slice(0, max_classes)
                      .filter((x) => x.count > pixel_count_thresh)

          let masks = sparse_classes.reduce((acc, idx, j) => {
            const present = classes.reduce((acc, x, i) => (x.idx === idx) ? i : acc, false)
            if (present !== false && class_probs[j] > this.state.softmax_threshold) {
              // acc.push(...idx_to_color(present))
              acc.push(...idx_to_color(idx))
              return acc
            }
            acc.push(...idx_to_color(80))
            return acc
          }, [])

          // classes = classes.map((x, i) => ({name: `${idx_to_category_name[x.idx]}`, color: colorString(idx_to_color(i))}))
          classes = classes.map((x, i) => ({name: `${idx_to_category_name[x.idx]}`, color: colorString(idx_to_color(x.idx))}))
          
          return [masks, classes]
      })
      masks = new Uint8ClampedArray(masks)
      masks = new ImageData(masks, 224, 224)
      this.setState({masks, classes})
    }
    this.debug.log(`num Tensors: ${tf.memory().numTensors}`)
  }

  render() {
    return (
      <div className="App">
        <Camera predict={this.predict} />
        <Results state={this.state} onSlideChange={this.updateSoftmaxThresh}/>
        <Info />
        <Debug debug={this.debug} logs={this.state.logs} paused={this.state.paused} />
      </div>
    )
  }
}

export default App;
