import React from 'react';
import './Camera.css'

class Camera extends React.Component {
  constructor(props) {
    super(props);
  }

  handleVideo = (stream) => {
    let video = document.querySelector("#cameraOutput");
    let viewportWidth = document.documentElement.getBoundingClientRect().width
    video.setAttribute('height', viewportWidth);
    video.srcObject = stream;
    // video.loadeddata = this.props.predict
  }

  componentDidMount() {
    var constraints = { video: { width: 1000, height: 1000 }, audio: false, };
    navigator.mediaDevices
      .getUserMedia(constraints)
      .then(this.handleVideo)
      .catch(console.error);
  }
  render() {
    return (
      <div className="Camera-container">
        <video autoPlay={true} id="cameraOutput"></video>
      </div>
    );
  }
}

export default Camera