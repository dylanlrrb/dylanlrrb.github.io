import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
// import AdvancedCamera from './components/AdvancedCamera/AdvancedCamera'
import Results from './components/Results/Results';
import Info from './components/Info/Info'
import Debug from './components/Debug/Debug';

const wait = (ms) => new Promise((resolve) => {
  setTimeout(() => {
    // console.log('waited', ms, 'ms')
    resolve()
  }, ms)
})

const range = (start, end, skip=1) => {
  return [...Array(end).keys()].filter(i => i % skip === 0 && i >= start)
}


class App extends React.Component {
  constructor(props) {
    super(props);
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_BEST-MSE/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_BEST-P/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_FINAL/model.json'

    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_proper_preprocess_BEST-P/model.json'

    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_ploss1_gram0-1_BEST-P/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_ploss1_gram0-1_BEST-MSE/model.json'

    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_ploss2_gram0-2_BEST-P/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_ploss2_gram0-2_BEST-MSE/model.json'
    
    this.radial_mask_memo = {}

    this.state = {
      loading: true,
      model: undefined,
      logs: [],
      paused: false,
      step: 0,
      originalImg: undefined,
      enhancedImg: undefined,
      offsetEnhancedImg: undefined,
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
    if (this.state.model) {
      this.preventInteraction(true)
      const model_output_dim = 224
      const upscale_factor = 4
      const num_side_tiles = Math.trunc((tensor.shape[0] * upscale_factor) / model_output_dim)
      const updacale_dim = num_side_tiles * model_output_dim
      let upscaledImg = tf.image.resizeBilinear(tensor, [updacale_dim,updacale_dim])

      const crops = tf.tidy(() => {
        const crops = []
        for (const h of range(0, num_side_tiles)) {
          for (const w of range(0, num_side_tiles)) {
            const crop = tf.slice(upscaledImg, [h*model_output_dim, w*model_output_dim, 0], [model_output_dim, model_output_dim, -1])
            crops.push(crop)
          }
        }
        return crops
      })

      const offsetCrops = tf.tidy(() => {
        const crops = []
        for (const h of range(0, num_side_tiles - 1)) {
          for (const w of range(0, num_side_tiles - 1)) {
            const crop = tf.slice(upscaledImg, [(h*model_output_dim) + (model_output_dim / 2), (w*model_output_dim) + (model_output_dim / 2), 0], [model_output_dim, model_output_dim, -1])
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
        let processedCrop = tf.squeeze(expandedPred)
        processedCrop = tf.concat([processedCrop, this.radial_mask(model_output_dim)], 2)
        console.log('patch:', i++, ', memory:', tf.memory())
        processedCrops.push(processedCrop)
        crop.dispose()
        expandedCrop.dispose()
        expandedPred.dispose()
      }

      const offsetProcessedCrops = []
      let j = 0
      for (const crop of offsetCrops) {
        const expandedCrop = tf.expandDims(crop, 0)
        await tf.nextFrame()
        const expandedPred = this.state.model.predict(expandedCrop)
        await wait(10)
        let processedCrop = tf.squeeze(expandedPred)
        processedCrop = tf.concat([processedCrop, this.radial_mask(model_output_dim)], 2)
        console.log('patch:', j++, ', memory:', tf.memory())
        offsetProcessedCrops.push(processedCrop)
        crop.dispose()
        expandedCrop.dispose()
        expandedPred.dispose()
      }

      console.log('FINISH PROCESSING', tf.memory())
      this.debug.log(`FINISH PROCESSING`)

      const rows = tf.tidy(() => {
        const rows = []
        for (const i of range(0, Math.pow(num_side_tiles, 2), num_side_tiles)) {
          let row = processedCrops[i]
          for (const crop of processedCrops.slice(i+1, i+num_side_tiles)) {
            row = tf.concat([row, crop], 1)
          }
          rows.push(row)
        }
        processedCrops.forEach(c => c.dispose())
        return rows
      })

      const offsetRows = tf.tidy(() => {
        const rows = []
        for (const i of range(0, Math.pow(num_side_tiles - 1, 2), num_side_tiles - 1)) {
          let row = offsetProcessedCrops[i]
          for (const crop of offsetProcessedCrops.slice(i+1, i+num_side_tiles - 1)) {
            row = tf.concat([row, crop], 1)
          }
          rows.push(row)
        }
        offsetProcessedCrops.forEach(c => c.dispose())
        return rows
      })

      console.log('FINISH BUILDING ROWS', tf.memory())
      this.debug.log(`FINISH BUILDING ROWS`)

      const enhancedImg = tf.tidy(() => {
        let whole = rows[0]
        for (const row of rows.slice(1)) {
          whole = tf.concat([whole, row], 0)
        }
        rows.forEach(r => r.dispose())
        return whole
      })

      const offsetEnhancedImg = tf.tidy(() => {
        let whole = offsetRows[0]
        for (const row of offsetRows.slice(1)) {
          whole = tf.concat([whole, row], 0)
        }
        offsetRows.forEach(r => r.dispose())
        return whole
      })

      console.log('FINISH BUILDING ENHANCED IMAGE', tf.memory())
      this.debug.log(`FINISH BUILDING ENHANCED IMAGE`)

      const original = tf.tidy(() => {
        const original = tf.concat([upscaledImg, tf.fill([updacale_dim, updacale_dim, 1], 255)], 2).toInt()
        return original
      })

      console.log('FINISH ADDING ALPHA TO ORIGINAL', tf.memory())
      this.debug.log(`FINISH ADDING ALPHA TO ORIGINAL`)

      let enhanced = tf.tidy(() => {
        let enhanced = tf.clipByValue(enhancedImg, -1, 1)
        enhanced = enhanced.add(tf.scalar(1))
        enhanced = tf.div(enhanced, tf.scalar(2))
        return enhanced
      })

      let offsetEnhanced = tf.tidy(() => {
        let enhanced = tf.clipByValue(offsetEnhancedImg, -1, 1)
        enhanced = enhanced.add(tf.scalar(1))
        enhanced = tf.div(enhanced, tf.scalar(2))
        // pad so the final offset image is padded with transperancy to same dimensions as other enhanced image
        enhanced = enhanced.pad([[112,112], [112,112], [0,0]])
        return enhanced
      })

      console.log('FINISH DENORMALIZING ENHANCED', tf.memory())
      this.debug.log(`FINISH DENORMALIZING ENHANCED`)
     
      let originalPixels = await tf.browser.toPixels(original)
      let enhancedPixels = await tf.browser.toPixels(enhanced)
      let offsetEnhancedPixels = await tf.browser.toPixels(offsetEnhanced)

      console.log('FINISH GETTING PIXELS', tf.memory())
      this.debug.log(`FINISH GETTING PIXELS`)

      let originalImageData = new ImageData(originalPixels, updacale_dim, updacale_dim)
      let enhancedImageData = new ImageData(enhancedPixels, updacale_dim, updacale_dim)
      let offsetEnhancedImageData = new ImageData(offsetEnhancedPixels, updacale_dim, updacale_dim)

      console.log('FINISH CONVERTING TO IMAGE DATA', tf.memory())
      this.debug.log(`FINISH CONVERTING TO IMAGE DATA`)

      tensor.dispose()
      enhancedImg.dispose()
      offsetEnhancedImg.dispose()
      upscaledImg.dispose()
      enhanced.dispose()
      offsetEnhanced.dispose()
      original.dispose()

      this.setState({
        step: 1,
        originalImg: originalImageData,
        enhancedImg: enhancedImageData,
        offsetEnhancedImg: offsetEnhancedImageData}, () => this.preventInteraction(false))
    }
  }

  radial_mask = (dim) => {
    if (this.radial_mask_memo[dim]) {
      return this.radial_mask_memo[dim]
    }
    const X = [[...range(0, dim)]]
    const Y = [...range(0, dim)].map((i => [i]))
    const center = [dim / 2, dim / 2]

    const a = (tf.sub(X, center[0])).pow(2)

    const b = (tf.sub(Y, center[1])).pow(2)

    let dist_from_center = tf.sqrt(tf.add(a, b))

    const max_dist = tf.max(dist_from_center).sub(2)

    // const max_dist = dim / 2
    // dist_from_center = dist_from_center.clipByValue(0, max_dist)
    

    dist_from_center = dist_from_center.sub(max_dist)
    dist_from_center = tf.abs(dist_from_center)
    dist_from_center = dist_from_center.div(max_dist)
    dist_from_center = tf.expandDims(dist_from_center, -1)
    // needed for tanh scaling
    dist_from_center = dist_from_center.mul(2)
    dist_from_center = dist_from_center.sub(1)
    this.radial_mask_memo[dim] = dist_from_center

    return dist_from_center
  }

  render() {
    return (
      <div className="App">
        {this.state.step === 0 ? <Camera enhance={this.enhance} preventInteraction={this.preventInteraction} debug={this.debug} /> : ''}
        {this.state.step === 1 ? <Results originalImg={this.state.originalImg} enhancedImg={this.state.enhancedImg} offsetEnhancedImg={this.state.offsetEnhancedImg} retake={this.retake} enhance={this.enhance} debug={this.debug} /> : ''}
        <Info />
        <Debug debug={this.debug} logs={this.state.logs} paused={this.state.paused} />
        {this.state.loading ? <div className="App-scrim"><div className='App-loader'></div>{this.state.model ? "Super Resolving Image..." : "Loading Model..."}</div> : null}
      </div>
    )
  }
}

export default App;
