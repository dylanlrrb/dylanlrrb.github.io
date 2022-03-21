import React from 'react';
import './Camera.css'
import camera_flip from '../icons/camera-flip.png';
import camera from '../icons/camera.png';
import * as tf from "@tensorflow/tfjs"

class Camera extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      camera: undefined,
      canvas: undefined,
      video:  undefined,
      facingMode: 'environment',
      loading: true,
      stopped: false,
      error: false,
    }
  }

  toggleFacingMode = () => {
    this.state.camera.stop()
    this.setState({loading: true})
    if(this.state.facingMode === 'environment'){
      this.setState({facingMode: 'user'}, this.setupCamera)
    } else {
      this.setState({facingMode: 'environment'}, this.setupCamera)
    }
  }

  setupCamera = async (stream) => {
    let video = document.querySelector("#cameraOutput");
    if (video) {
      let viewportWidth = document.querySelector('#root').getBoundingClientRect().width
      video.setAttribute('height', viewportWidth);
      video.setAttribute('width', viewportWidth);
  
      let canvas = document.querySelector("canvas")
      let camera = await tf.data.webcam(video, {facingMode: this.state.facingMode});
      camera.start()
  
      this.setState({
        camera,
        canvas,
        video,
        loading: false
      })
    }
  }

  retake = async () => {
    this.setState({loading: true,
      stopped: false,
    })
    await this.state.camera.start()
    this.state.canvas.classList.add('display-none')
    this.state.video.classList.remove('display-none')

    this.setState({loading: false})
  }

  captureFrame = async () => {
    let image = await this.state.camera.capture();
    tf.browser.toPixels(image, this.state.canvas);
    this.state.camera.stop()
    this.state.video.classList.add('display-none')
    this.state.canvas.classList.remove('display-none')
    this.setState({stopped: true})
    this.props.predict(image)
  }

  predict = async () => {
    if (!this.state.stopped){
      window.requestAnimationFrame(async () => {
        let image = await this.state.camera.capture();
        this.props.predict(image)
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
      return <span className='position-absolute'>There was an error starting the webcam</span>
    } else if (this.state.loading) {
      // return <span className='position-absolute'>XXXX</span>
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
            <video autoPlay={true} id="cameraOutput"></video>
            <canvas className='display-none'></canvas>
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