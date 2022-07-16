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
        <h2>Super Resolution</h2>
        <p>This demo super resolves an image with a GAN trined on the <a href="https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset" target="_blank" rel="noopener noreferrer">Flickr30k dataset</a></p>
        <p>Take a photo and zoom in by pinching and dragging the image. Use the slider to reveal the super resolved image on the left for comparison</p>
        <p>The GAN was trained with a custom Perceptual loss function, more info can be found in the project reop <a href="https://github.com/dylanlrrb/dylanlrrb.github.io/tree/master/portfolio/super_resolution" target="_blank"  rel="noopener noreferrer">on Github</a></p>
      </div>
    </div>
    
  </div>)
}

export default Info
