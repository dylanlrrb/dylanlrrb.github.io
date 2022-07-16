import React from 'react';
import './AdvancedCamera.css'
import camera_flip from './icons/camera-flip.png';
import camera from './icons/camera.png';
import * as tf from "@tensorflow/tfjs"

function stopStreamedVideo(videoElem) {
  const stream = videoElem.srcObject;
  const tracks = stream.getTracks();

  tracks.forEach(function(track) {
    track.stop();
  });

  videoElem.srcObject = null;
}

const BlobToImageData = function(blob){
  let blobUrl = URL.createObjectURL(blob);

  return new Promise((resolve, reject) => {
              let img = new Image();
              img.onload = () => resolve(img);
              img.onerror = err => reject(err);
              img.src = blobUrl;
          }).then(img => {
              URL.revokeObjectURL(blobUrl);
              // Limit to 256x256px while preserving aspect ratio
              let [w,h] = [img.width,img.height]
              // let aspectRatio = w/h
              // // Say the file is 1920x1080
              // // divide max(w,h) by 256 to get factor
              // let factor = Math.max(w,h)/256
              // w = w/factor
              // h = h/factor

              // REMINDER
              // 256x256 = 65536 pixels with 4 channels (RGBA) = 262144 data points for each image
              // Data is encoded as Uint8ClampedArray with BYTES_PER_ELEMENT = 1
              // So each images = 262144bytes
              // 1000 images = 260Mb
              let canvas = document.createElement("canvas");
              canvas.width = w;
              canvas.height = h;
              let ctx = canvas.getContext("2d");
              ctx.drawImage(img, 0, 0);

              return ctx.getImageData(0, 0, w, h);    // some browsers synchronously decode image here
          })
}

class Camera extends React.Component {
  constructor(props) {
    super(props);
    this.waiting = false
    this.state = {
      mediaStreamTrack: undefined,
      imageCapture: undefined,
      webcam: undefined,
      canvas: undefined,
      viewportWidth: undefined,
      capabilities: undefined,
      settings: undefined,
      facingMode: 'environment',
      loading: true,
      stopped: false,
      error: undefined,
    }
  }

  toggleFacingMode = () => {
    stopStreamedVideo(this.state.webcam)
    this.props.debug.log(`state's facing mode: ${this.state.facingMode}`)
    if (this.state.facingMode === 'environment') {
      this.setState({facingMode: 'user'}, this.setupCamera)
    } else {
      this.setState({facingMode: 'environment'}, this.setupCamera)
    }
  }

  setupCamera = () => {
    setTimeout(async () => {
      let viewportWidth = Math.floor(document.querySelector('.App').getBoundingClientRect().width)
      let webcam = document.querySelector('#webcam')
      const canvas = document.querySelector('.Advanced-Camera-container canvas')
      const mediaStream = await navigator.mediaDevices.getUserMedia({audio: false, video: { facingMode: this.state.facingMode }})
      webcam.srcObject = mediaStream

      const mediaStreamTrack = mediaStream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(mediaStreamTrack);
      const capabilities = mediaStreamTrack.getCapabilities()
      const settings = mediaStreamTrack.getSettings()
      const constraints = mediaStreamTrack.getConstraints()

      
      console.log('capabilities', capabilities)
      console.log('settings', settings)
      console.log('constraints', constraints)
      this.props.debug.log(`capabilities: ${Object.keys(capabilities)}`)
      this.props.debug.log(`settings: ${Object.keys(settings)}`)
      this.props.debug.log(`constraints: ${JSON.stringify(constraints)}`)
      // this.props.debug.log(`state's facing mode: ${this.state.facingMode}`)

      this.setState({
        mediaStreamTrack,
        imageCapture,
        webcam,
        canvas,
        viewportWidth,
        capabilities,
        settings,
        loading: false,
        stopped: false,
      })

    }, 10)
  }

  retake = async () => {
    this.setState({loading: true,
      stopped: false,
    }, this.setupCamera)
  }

  captureFrame = async () => {
    this.setState({loading: true})
    const blob = await this.state.imageCapture.takePhoto()
    stopStreamedVideo(this.state.webcam)
    const imagebitmap = await BlobToImageData(blob)
    
    // printing the captured image size
    console.log(imagebitmap.height, imagebitmap.width)
    this.props.debug.log(`height: ${imagebitmap.height}, width: ${imagebitmap.width}`)
    this.state.canvas.height =  imagebitmap.height
    this.state.canvas.width =  imagebitmap.width
    const ctx = this.state.canvas.getContext('2d')
    ctx.putImageData(imagebitmap, 0, 0)
    this.props.debug.log('one')

    this.setState({stopped: true})

    this.props.preventInteraction(true)
    this.props.debug.log('two')
    const tensor = await tf.browser.fromPixels(imagebitmap)
    this.props.debug.log(`${tensor.shape}`)
    await this.props.enhance(tensor)
    this.props.debug.log('three')
    this.props.preventInteraction(false)
  }

  componentDidMount() {
    this.setupCamera()
  }

  componentWillUnmount() {
    // stop the stream
  }

  videoMessage = () => {
    if (this.state.error) {
      return <span className='position-absolute'>{this.state.error}</span>
    } else if (this.state.loading) {
      return <div className='Advanced-Camera-scrim'><div className="Camera-loader position-absolute"></div></div>
    }
  }

  render() {
    return (
      <div>
        <button className='Advanced-Camera-toggle' onClick={this.toggleFacingMode}><img src={camera_flip} alt="" /></button>
        <div className="Advanced-Camera-container">
          {this.videoMessage()}
          <div>
            <video autoPlay={true} className={this.state.stopped ? 'height-0' : ''} id="webcam"></video>
            <canvas className={this.state.stopped ? '' : 'height-0'}></canvas>
          </div>
        </div>
        <div className='Advanced-Camera-button'>
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