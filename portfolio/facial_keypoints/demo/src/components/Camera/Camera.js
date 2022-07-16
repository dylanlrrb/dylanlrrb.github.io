import React from 'react';
import './Camera.css'
import camera_flip from './icons/camera-flip.png';
import camera from './icons/camera.png';
import * as tf from "@tensorflow/tfjs"

class Camera extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      camera: undefined,
      canvas: undefined,
      facingMode: 'environment',
      loading: true,
      stopped: false,
      error: undefined,
    }
  }

  toggleFacingMode = () => {
    this.state.camera.stop()
    this.setState({loading: true, error: undefined})
    if(this.state.facingMode === 'environment'){
      this.setState({facingMode: 'user', camera: undefined}, this.setupCamera)
    } else {
      this.setState({facingMode: 'environment', camera: undefined}, this.setupCamera)
    }
  }

  setupCamera = () => {
    setTimeout(async () => {
      let viewportWidth = Math.floor(document.querySelector('.App').getBoundingClientRect().width)
      let canvas = document.querySelector('#webcam')
      // canvas.setAttribute('height', viewportWidth)
      // canvas.setAttribute('width', viewportWidth)
      let camera = await tf.data
                          .webcam(null, {
                            facingMode: this.state.facingMode,
                            resizeHeight: viewportWidth,
                            resizeWidth: viewportWidth
                          })
                          .catch((e) => {
                            this.setState({error: `There was an error initalizing the webcam:\n ${e}`})
                          });
      await camera.start().catch((e) => {
        this.setState({error: `There was an error starting the webcam:\n ${e}`})
      })

      return this.setState({
        camera,
        canvas,
        loading: false,
        stopped: false,
      }, this.tick)
    }, 10)
  }

  retake = async () => {
    this.setState({loading: true,
      stopped: false,
    })
    await this.state.camera.start()
    this.setState({loading: false}, this.tick)
  }

  captureFrame = async () => {
    let image = await this.state.camera.capture();
    this.state.camera.stop()
    this.setState({stopped: true})
    this.props.preventInteraction(true)
    await this.props.enhance(image)
    this.props.preventInteraction(false)
  }

  tick = () => {
    if (!this.state.stopped && this.state.camera){
      window.requestAnimationFrame(async () => {
        let image = await this.state.camera.capture();
        if (image) {
          await tf.browser.toPixels(image, this.state.canvas);
          this.props.predict(image)
          image.dispose()
        }
        this.props.debug.log(`num Tensors: ${tf.memory().numTensors}`)
        this.tick()
      })
    }
  }

  componentDidMount() {
    this.setupCamera()
  }

  componentWillUnmount() {
    this.state.camera.stop()
  }

  render() {
    return (
      <div>
        <button className='Camera-toggle' onClick={this.toggleFacingMode}><img src={camera_flip} alt="" /></button>
        <div className="Camera-container">
          <div>
            <canvas id="webcam"></canvas>
          </div>
        </div>
      </div>
    );
  }
}

export default Camera