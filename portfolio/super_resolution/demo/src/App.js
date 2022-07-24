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
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in256_4Xzoom_plossX0-1_iteration_12719/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in256_4Xzoom_plossX0-1_full_train_iteration_50839/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in256_4Xzoom_plossX0-1_full_train_iteration_24148/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in512_4Xzoom_plossX0-1_iteration_12719/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in256_4Xzoom_plossX0-1_full_train_iteration_48297/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in224_4Xzoom_plossX0-1_monilenet_backbone_iteration_9999/model.json'

    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/gan_in224_4Xzoom_plossX0-1_monilenet_backbone_conv5_lr_2e-4_iteration_17499/model.json' // best looking so far
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/4xzoom_mobilenet/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/4xzoom_mobilenet-mse-test/model.json'

    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_BEST-MSE/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_BEST-P/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_FINAL/model.json'
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/super_resolution/mobile_unet_proper_preprocess_BEST-P/model.json'
    
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
        processedCrop = tf.concat([processedCrop, this.radial_mask(224)], 2)
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
        processedCrop = tf.concat([processedCrop, this.radial_mask(224)], 2)
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

      this.setState({
        step: 1,
        originalImg: originalImageData,
        enhancedImg: enhancedImageData,
        offsetEnhancedImg: offsetEnhancedImageData})

      tensor.dispose()
      enhancedImg.dispose()
      offsetEnhancedImg.dispose()
      upscaledImg.dispose()
      enhanced.dispose()
      offsetEnhanced.dispose()
      original.dispose()

    }
  }

  radial_mask = (updacale_dim) => {
    if (this.radial_mask_memo[updacale_dim]) {
      return this.radial_mask_memo[updacale_dim]
    }
    const X = [[...range(0, updacale_dim)]]
    const Y = [...range(0, updacale_dim)].map((i => [i]))
    const center = [updacale_dim / 2, updacale_dim / 2]

    const a = (tf.sub(X, center[0])).pow(2)

    const b = (tf.sub(Y, center[1])).pow(2)

    let dist_from_center = tf.sqrt(tf.add(a, b))

    const max_dist = tf.max(dist_from_center)

    dist_from_center = dist_from_center.sub(max_dist)
    dist_from_center = tf.abs(dist_from_center)
    dist_from_center = dist_from_center.div(max_dist)
    dist_from_center = tf.expandDims(dist_from_center, -1)
    this.radial_mask_memo[updacale_dim] = dist_from_center

    return dist_from_center
  }

  render() {
    return (
      <div className="App">
        {this.state.step === 0 ? <Camera enhance={this.enhance} preventInteraction={this.preventInteraction} debug={this.debug} /> : ''}
        {this.state.step === 1 ? <Results originalImg={this.state.originalImg} enhancedImg={this.state.enhancedImg} offsetEnhancedImg={this.state.offsetEnhancedImg} retake={this.retake} debug={this.debug} /> : ''}
        <Info />
        <Debug debug={this.debug} logs={this.state.logs} paused={this.state.paused} />
        {this.state.loading ? <div className="App-scrim"><div className='App-loader'></div>{this.state.model ? "Super Resolving Image..." : "Loading Model..."}</div> : null}
      </div>
    )
  }
}

export default App;
