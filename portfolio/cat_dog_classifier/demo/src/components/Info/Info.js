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
        <h2>Cat/Dog Classifier</h2>
        <p>This demo predicts the type and breed of animal from a photo. App will predict based on the camera input in real time, or you can hit the Capture button to take a picture and make a prediction.</p>
        <p>The model was trained with tensorflow on cat and dog datasets and converted to a tensorflowjs graph model to be used for inference in the browser</p>
        <p>More info along with the source code for this project can be found <a href="https://github.com/dylanlrrb/dylanlrrb.github.io/tree/master/portfolio/cat_dog_classifier" target="_blank">on Github</a></p>
      </div>
    </div>
    
  </div>)
}

export default Info
