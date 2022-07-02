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
        <h2>Convolution Visualizer</h2>
        <p>This demo visualizes how a selection of convolutional layers from a trained <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet" target="_blank" rel="noopener noreferrer">Mobilenet</a> extracts features from images taken from live video</p>
        <p>You can see how, in higher layers, the filters of the CNN act rudementrey edge detectors and color extractors. Then in lower layers you the features extracted from the image become more abstract, blurrier, and temporatly stable as it finds patterns within patterns</p>
        <p>Looking at it another way, a bright activated pixel in the filter of a high layer will represent someing simple like a blue corner or a dark edge. While a bright activated pixel in a filter of a lower layer will represent something much more abstract and complicated has been detected, such as an eye or a feather for example.</p>
        <p>More info about this project's implementation can be found <a href="https://github.com/dylanlrrb/dylanlrrb.github.io/tree/master/portfolio/conv_visualizer" target="_blank" rel="noopener noreferrer">on Github</a></p>
      </div>
    </div>
    
  </div>)
}

export default Info
