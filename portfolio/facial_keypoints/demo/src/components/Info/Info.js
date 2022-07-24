import React, {useState} from "react";
import './Info.css';
import q_mark from './icons/q-mark.png';

let wait = false

const Info = () => {
  const [infoOpenState, setInfoOpenState] = useState(false)
  const toggle = () => {
    if (!wait) {
      setInfoOpenState(!infoOpenState)
      wait = true;
      setTimeout(() => wait = false, 200)
    }
  }
  return(<div className="Info">
    <div onClick={toggle} onTouchStart={toggle} className={`Info-scrim ${infoOpenState ? '' : 'display-none'}`}></div>
    <button className="Info-button" onClick={toggle}><img src={q_mark} alt="" /></button>
    <div className={`Info-modal ${infoOpenState ? '' : 'closed'}`}>
      <div onClick={toggle} onTouchStart={toggle} className="Info-handle"></div>
      <div className="Info-content">
        <h2>Facial Keypoints</h2>
        <p>This demo plots 68 facial landmarks from a live webcam feed using a MobileNetV2 CNN trained on the <a href="https://www.cs.tau.ac.il/~wolf/ytfaces/" target="_blank" rel="noopener noreferrer">YouTube Faces dataset</a></p>
        <p>To improve accuracy and standardize inputs to the model, a Haar Cascade trained to detect face bounding boxes is used to create image crops for the model to map keypoints on. This also allows multiple faces to be mapped at the same time.</p>
        <p>More info can be found in the project reop <a href="https://github.com/dylanlrrb/dylanlrrb.github.io/tree/master/portfolio/facial_keypoints" target="_blank"  rel="noopener noreferrer">on Github</a></p>
      </div>
    </div>
    
  </div>)
}

export default Info
