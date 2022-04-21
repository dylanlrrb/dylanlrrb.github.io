import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
import Results from './components/Results/Results';
import Info from './components/Info/Info'
import Debug from './components/Debug/Debug';


class App extends React.Component {
  constructor(props) {
    super(props);
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/input_224_6_blocks_perceptual_loss_full_train_1/model.json'
    this.state = {
      loading: true,
      model: undefined,
      logs: [],
      paused: false,
      step: 0,
      originalImg: undefined,
      enhancedImg: undefined,
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

  preventInteraction = (state) => {
    this.setState({loading: state})
  }

  retake = () => {
    this.setState({step: 0})
  }

  enhance = async (tensor) => {

    console.log('initial memory state', tf.memory())

    if (this.state.model) {
      const originalImgDim = tensor.shape[0]
      const upscale_factor = 4
      let newDim = originalImgDim * upscale_factor
      newDim = Math.floor(newDim / 224) * 224
      const tiles = Math.floor(newDim / 224)
      const originalImg = tf.image.resizeBilinear(tensor, [newDim,newDim])
      
      const crops = tf.tidy(() => {
        const crops = []
        for (const h of Array(tiles).keys()) {
          for (const w of Array(tiles).keys()) {
            crops.push(tf.slice3d(originalImg, [h*224, w*224, 0], [224, 224, 3]))
          }
        }
        return crops
      })

      console.log('FINISH CROPPING', tf.memory())

      // const processedCrops = tf.tidy(() => {
      //   return tf.unstack(this.state.model.predict(tf.div(tf.stack(crops), 255)), 0)
      // })

      const processedCrops = tf.tidy(() => {
        const processedCrops = []
        let i = 0
        for (const crop of crops) {
          const scaledCrop = tf.div(crop, 255)
          const expandedCrop = tf.expandDims(scaledCrop, 0)
          const expandedPred = this.state.model.predict(expandedCrop)
          const processedCrop = tf.squeeze(expandedPred)
          console.log('patch:', i++, ', tensors:', tf.memory())
          processedCrops.push(processedCrop)
          crop.dispose()
          expandedCrop.dispose()
          expandedPred.dispose()
        }
        return processedCrops
      })

      console.log('FINISH PROCESSING', tf.memory())
      
      const rows = tf.tidy(() => {
        const rows = []
        for (const t of [...Array(tiles*tiles).keys()].filter(i => i % tiles === 0)) {
          rows.push(tf.concat(processedCrops.slice(t, t+tiles), 1))
        }
        processedCrops.forEach(c => c.dispose())
        return rows
      })

      console.log('FINISH BUILDING ROWS', tf.memory())
      
      const enhancedImg = tf.tidy(() => {
        const enhancedImg = tf.clipByValue(tf.concat(rows, 0), 0, 1)
        rows.forEach(r => r.dispose())
        return enhancedImg
      })

      console.log('FINISH BUILDING ENHANCED IMAGE', tf.memory())
      
      const original = tf.tidy(() => {
        const original = tf.concat([originalImg, tf.fill([newDim, newDim, 1], 255)], 2).toInt()
        return original
      }) 

      console.log('FINISH ADDING ALPHA TO ORIGINAL', tf.memory())
     
      const enhanced = tf.tidy(() => {
        const enhanced = tf.concat([enhancedImg, tf.fill([newDim, newDim, 1], 1)], 2)
        return enhanced
      })

      console.log('FINISH ADDING ALPHA TO ENHANCED', tf.memory())
    

      let originalPixels = await tf.browser.toPixels(original)
      let enhancedPixels = await tf.browser.toPixels(enhanced)

      console.log('FINISH GETTING PIXELS', tf.memory())

      let originalImageData = new ImageData(originalPixels, newDim, newDim)
      let enhancedImageData = new ImageData(enhancedPixels, newDim, newDim)

      console.log('FINISH CONVERTING TO IMAGE DATA', tf.memory())

      this.setState({step: 1, originalImg: originalImageData, enhancedImg: enhancedImageData})
      // this.setState({step: 1})

      // figure out what tensors you need to clean up here
      tensor.dispose()
      enhancedImg.dispose()
      originalImg.dispose()
      enhanced.dispose()
      original.dispose()
      // crops.forEach(c => c.dispose())
      // processedCrops.forEach(c => c.dispose())
      // rows.forEach(r => r.dispose())
    }
    
  }

  render() {
    return (
      <div className="App">
        {this.state.step === 0 ? <Camera enhance={this.enhance} preventInteraction={this.preventInteraction} debug={this.debug} /> : ''}
        {this.state.step === 1 ? <Results originalImg={this.state.originalImg} enhancedImg={this.state.enhancedImg} retake={this.retake} debug={this.debug} /> : ''}
        <Info />
        <Debug debug={this.debug} logs={this.state.logs} paused={this.state.paused} />
        {this.state.loading ? <div className="App-scrim"><div className='App-loader'></div></div> : null}
      </div>
    )
  }
}

export default App;
