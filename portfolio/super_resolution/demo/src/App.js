import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
import AdvancedCamera from './components/AdvancedCamera/AdvancedCamera'
import Results from './components/Results/Results';
import Info from './components/Info/Info'
import Debug from './components/Debug/Debug';

const wait = (ms) => new Promise((resolve) => {
  setTimeout(() => {
    console.log('waited', ms, 'ms')
    resolve()
  }, ms)
})

const range = (start, end, skip=1) => {
  return [...Array(end).keys()].filter(i => i % skip === 0 && i >= start)
}


class App extends React.Component {
  constructor(props) {
    super(props);
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in256_4Xzoom_plossX0-1_iteration_12719/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in512_4Xzoom_plossX0-1_iteration_12719/model.json'
    
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
    this.debug.log(`initial memory state`)

    if (this.state.model) {
      this.debug.log(`inside model block`)
      const model_output_dim = 256
      const upscale_factor = 4
      let overlap = 15
      const trim = 10
      
      const minSizeTensor = tf.image.resizeBilinear(tensor, [model_output_dim,model_output_dim])
      const trimmed_model_output_dim = model_output_dim - trim
      const originalImgDim = minSizeTensor.shape[0]
      const numTiles = Math.floor(originalImgDim / trimmed_model_output_dim)
      const upscaledNumTiles = numTiles * upscale_factor
      const upscaledDim = upscaledNumTiles*model_output_dim + (overlap * (upscaledNumTiles - 1))
      this.debug.log(`before resized`)
      let upscaledImg = tf.image.resizeBilinear(minSizeTensor, [upscaledDim,upscaledDim])
      this.debug.log(`resized`)
      
      const crops = tf.tidy(() => {
        const crops = []
        for (const h of range(0, upscaledNumTiles)) {
          for (const w of range(0, upscaledNumTiles)) {
            const crop = tf.slice(upscaledImg, [Math.max(h*model_output_dim - h*overlap, 0), Math.max(w*model_output_dim - w*overlap, 0), 0], [model_output_dim, model_output_dim, -1])
            crops.push(crop)
          }
        }
        return crops
      })

      console.log('FINISH CROPPING', tf.memory())
      this.debug.log(`FINISH CROPPING`)

      const processedCrops = []
      let i = 0
      for (const crop of crops) {
        const expandedCrop = tf.expandDims(crop, 0)
        await tf.nextFrame()
        const expandedPred = this.state.model.predict(expandedCrop)
        await wait(10)
        const processedCrop = tf.squeeze(expandedPred)
        const trimmedProcessedCrop = tf.slice(processedCrop, [trim, trim, 0], [trimmed_model_output_dim, trimmed_model_output_dim, -1])
        console.log('patch:', i++, ', memory:', tf.memory())
        processedCrops.push(trimmedProcessedCrop)
        crop.dispose()
        expandedCrop.dispose()
        expandedPred.dispose()
        processedCrop.dispose()
      }

      console.log('FINISH PROCESSING', tf.memory())
      this.debug.log(`FINISH PROCESSING`)

      overlap = overlap - trim
      
      const rows = tf.tidy(() => {
        const rows = []
        for (const i of range(0, upscaledNumTiles*upscaledNumTiles, upscaledNumTiles)) {
          let row = processedCrops[i]
          for (const crop of processedCrops.slice(i+1, i+upscaledNumTiles)) {
            const left = tf.slice(row, [0, 0, 0], [-1, row.shape[1] - overlap, -1])
            const right = tf.slice(crop, [0, overlap, 0], [-1, -1, -1])
            const overlap_left = tf.slice(row, [0, row.shape[1] - overlap, 0], [-1, -1, -1])
            const overlap_right = tf.slice(crop, [0, 0, 0], [-1, overlap, -1])
            const temp = overlap_left.add(overlap_right)
            const overlap_avg = tf.div(temp, 2)
            row = tf.concat([left, overlap_avg, right], 1)
          }
          rows.push(row)
        }
        processedCrops.forEach(c => c.dispose())
        return rows
      })

      console.log('FINISH BUILDING ROWS', tf.memory())
      this.debug.log(`FINISH BUILDING ROWS`)
      
      const enhancedImg = tf.tidy(() => {
        let whole = rows[0]
        for (const row of rows.slice(1)) {
          const top = tf.slice(whole, [0, 0, 0], [whole.shape[0] - overlap, -1, -1])
          const bottom = tf.slice(row, [overlap, 0, 0], [-1, -1, -1])
          const overlap_top = tf.slice(whole, [whole.shape[0] - overlap, 0, 0], [-1, -1, -1])
          const overlap_bottom = tf.slice(row, [0, 0, 0], [overlap, -1, -1])
          const overlap_avg = tf.div(overlap_top.add(overlap_bottom), 2)
          whole = tf.concat([top, overlap_avg, bottom], 0)
        }
        rows.forEach(r => r.dispose())
        whole = tf.clipByValue(whole, -1, 1)
        whole = whole.add(tf.scalar(1))
        whole = tf.div(whole, tf.scalar(2))
        return whole
      })

      console.log('FINISH BUILDING ENHANCED IMAGE', tf.memory())
      this.debug.log(`FINISH BUILDING ENHANCED IMAGE`)

      console.log('upscaledImg shape', upscaledImg.shape) 
      console.log('enhancedImg shape', enhancedImg.shape) 

      const enhancedImgDim = enhancedImg.shape[0]

      upscaledImg = tf.slice(upscaledImg, [trim, trim, 0], [enhancedImgDim, enhancedImgDim, -1])
      const original = tf.tidy(() => {
        const original = tf.concat([upscaledImg, tf.fill([enhancedImgDim, enhancedImgDim, 1], 255)], 2).toInt()
        return original
      }) 

      console.log('upscaledImg shape AFTER', upscaledImg.shape) 

      console.log('FINISH ADDING ALPHA TO ORIGINAL', tf.memory())
      this.debug.log(`FINISH ADDING ALPHA TO ORIGINAL`)
     
      let enhanced = tf.tidy(() => {
        const enhanced = tf.concat([enhancedImg, tf.fill([enhancedImgDim, enhancedImgDim, 1], 1)], 2)
        return enhanced
      })
      // enhanced = tf.slice(upscaledImg, [0, 0, 0], [enhancedImgDim, enhancedImgDim, -1])

      console.log('FINISH ADDING ALPHA TO ENHANCED', tf.memory())
      this.debug.log(`FINISH ADDING ALPHA TO ENHANCED`)
    

      let originalPixels = await tf.browser.toPixels(original)
      let enhancedPixels = await tf.browser.toPixels(enhanced)

      console.log('FINISH GETTING PIXELS', tf.memory())
      this.debug.log(`FINISH GETTING PIXELS`)

      let originalImageData = new ImageData(originalPixels, enhancedImgDim, enhancedImgDim)
      let enhancedImageData = new ImageData(enhancedPixels, enhancedImgDim, enhancedImgDim)

      console.log('FINISH CONVERTING TO IMAGE DATA', tf.memory())
      this.debug.log(`FINISH CONVERTING TO IMAGE DATA`)

      this.setState({step: 1, originalImg: originalImageData, enhancedImg: enhancedImageData})

      tensor.dispose()
      minSizeTensor.dispose()
      enhancedImg.dispose()
      upscaledImg.dispose()
      enhanced.dispose()
      original.dispose()

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
