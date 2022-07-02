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
        <h2>Common Object Image Segmentation</h2>
        <p>This demo segments live video by attempting to assgn a class to each pixel and adding a colored overlay</p>
        <p>The segmentation model is a U-Net with a MobilenetV2 model used as the downsample path. It was trained on the <a href="https://cocodataset.org/#home" target="_blank" rel="noopener noreferrer">COCO Dataset</a> to recognize and classify 80 diferent classes of common objects</p>
        <p>More details on implementation can be found in the project repo <a href="https://github.com/dylanlrrb/dylanlrrb.github.io/tree/master/portfolio/image_segmentation" target="_blank" rel="noopener noreferrer">on Github</a></p>
      </div>
    </div>
    
  </div>)
}

export default Info
