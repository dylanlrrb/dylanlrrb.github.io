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

  setupCamera = (stream) => {
    setTimeout(async () => {
      let viewportWidth = Math.floor(document.querySelector('.App').getBoundingClientRect().width)
      let canvas = document.querySelector('#webcam')
      canvas.setAttribute('height', viewportWidth)
      canvas.setAttribute('width', viewportWidth)
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
        // loading: false,
        stopped: false,
      }, this.tick)
    }, 10)
  }

  retake = async () => {
    this.setState({loading: true,
      stopped: false,
    })
    await this.state.camera.start()
    this.tick()
    // this.setState({loading: false}, this.tick)
  }

  captureFrame = async () => {
    let image = await this.state.camera.capture();
    await tf.browser.toPixels(image, this.state.canvas);
    this.state.camera.stop()
    this.setState({stopped: true})
    this.props.predict(image)
  }

  tick = () => {
    if (!this.state.stopped && this.state.camera){
      window.requestAnimationFrame(async () => {
        let image = await this.state.camera.capture();
        if (image) {
          await tf.browser.toPixels(image, this.state.canvas);
          this.props.predict(image)
          image.dispose()
          if (this.state.loading === true) {
            this.setState({loading: false})
          }
        }
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

  videoMessage = () => {
    if (this.state.error) {
      return <span className='position-absolute'>{this.state.error}</span>
    } else if (this.state.loading) {
      return <div className='Camera-scrim'><div className="Camera-loader position-absolute"></div></div>
    }
  }

  render() {
    return (
      <div>
        <button className='Camera-toggle' onClick={this.toggleFacingMode}><img src={camera_flip} alt="" /></button>
        <div className="Camera-container">
          {this.videoMessage()}
          <div>
            <canvas id="webcam"></canvas>
          </div>
        </div>
        <div className='Camera-button'>
          {this.state.stopped
            ? <button onClick={this.retake}><h2>Retake</h2></button> 
            : <button disabled={this.state.loading} onClick={this.captureFrame}><img src={camera} alt="" /></button>
          }
        </div>
        
      </div>
    );
  }
}

export default Camera