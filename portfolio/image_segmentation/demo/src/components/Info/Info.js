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
      </div>
    </div>
    
  </div>)
}

export default Info
