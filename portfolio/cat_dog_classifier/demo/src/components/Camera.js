import React from 'react';
import './Camera.css'
import camera_flip from '../icons/camera-flip.png';
import camera from '../icons/camera.png';
import * as tf from "@tensorflow/tfjs"

let throttlePause;
 
const throttle = (callback, waitCallback, time) => {
  //don't run the function if throttlePause is true
  if (throttlePause) return waitCallback();
 
  //set throttlePause to true after the if condition. This allows the function to be run once
  throttlePause = true;
   
  //setTimeout runs the callback within the specified time
  setTimeout(() => {
    callback();
     
    //throttlePause is set to false once the function has been called, allowing the throttle function to loop
    throttlePause = false;
  }, time);
};

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

  setupCamera = async (stream) => {
    let viewportWidth = Math.floor(document.querySelector('#root').getBoundingClientRect().width)
    let canvas = document.querySelector('#webcam')
    canvas.setAttribute('height', viewportWidth)
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
    await tf.browser.toPixels(image, this.state.canvas);
    this.state.camera.stop()
    this.setState({stopped: true})
    this.props.predict(image)
  }

  tick = async () => {
    if (!this.state.stopped && this.state.camera){
      window.requestAnimationFrame(async () => {
        let image = await this.state.camera.capture().catch(()=>{});
        if (image) {
          await tf.browser.toPixels(image, this.state.canvas);
          // throttle(() => this.props.predict(image), () => image.dispose(), 100)
          this.props.predict(image)
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