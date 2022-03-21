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
      error: undefined,
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
    if (!video) {return this.setState({error: 'Video element is missing'})}
    let canvas = document.querySelector("canvas")
    if (!canvas) {return this.setState({error: 'Canvas element is missing'})}
    let viewportWidth = document.querySelector('#root').getBoundingClientRect().width
    video.setAttribute('height', viewportWidth);
    video.setAttribute('width', viewportWidth);

    let camera = await tf.data
                        .webcam(video, {facingMode: this.state.facingMode})
                        .catch((e) => {
                          this.setState({error: `There was an error initalizing the webcam:\n ${e}`})
                        });
    await camera.start().catch((e) => {
      this.setState({error: `There was an error starting the webcam:\n ${e}`})
    })

    return this.setState({
      camera,
      canvas,
      video,
      loading: false
    })
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
    // this.props.predict(image)
  }

  tick = async () => {
    if (!this.state.stopped){
      window.requestAnimationFrame(async () => {
        let image = await this.state.camera.capture();
        // this.props.predict(image)
        // possibly draw frames on the canvas rather than showing the video element if the predicted overlay is lagging behind the live camera 
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