import React, {useState} from "react";
import './Info.css';
import q_mark from '../icons/q-mark.png';

const Info = () => {
  const [infoOpenState, setInfoOpenState] = useState(false)
  const toggle = () => setInfoOpenState(!infoOpenState)
  return(<div>
    <div onClick={toggle} onTouchStart={toggle} className={`Info-scrim ${infoOpenState ? '' : 'display-none'}`}></div>
    <button className="Info-button" onClick={toggle}><img src={q_mark} alt="" /></button>
    <div onClick={toggle} onTouchStart={toggle} className={`Info-modal ${infoOpenState ? '' : 'closed'}`}>
      <p>This is some info about the <a href="https://google.com">demo</a></p>
      <p>This is some info about the <a href="https://google.com">demo</a></p>
      <p>This is some info about the <a href="https://google.com">demo</a></p>
      <p>This is some info about the <a href="https://google.com">demo</a></p>
    </div>
  </div>)
}

export default Info
