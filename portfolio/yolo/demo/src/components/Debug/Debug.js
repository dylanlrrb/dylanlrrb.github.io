import React, {useState} from "react";
import './Debug.css'
import debug_icon from './icons/debug.png';

const Debug = ({debug, logs, paused}) => {
  const [debugHiddenState, setdebugHiddenState] = useState(true)

  const [hiddenTapState, setHiddenTapState] = useState(1)
  const hiddenTap = () => {
    if (hiddenTapState >= 5 ) {
      setdebugHiddenState(false)
    }
    setHiddenTapState(hiddenTapState + 1)
  }

  const [debugOpenState, setDebugOpenState] = useState(false)
  const toggleOpen = () => {
    if (!debugHiddenState) {
      setDebugOpenState(!debugOpenState)
    }
  }

  const onClick = () => {
    hiddenTap()
    toggleOpen()
  }

  return(<div>
    <button className={`Debug-icon ${debugHiddenState ? 'hidden' : ''}`} onClick={onClick}>
      <img  src={debug_icon} alt="" />
    </button>
    <div className={`Debug ${debugOpenState ? '' : 'width-0'}`}>
      <ul className='Debug-list'>
        {logs.map((log, index)=>{
          return log.type === 'error' 
            ?  <li className='Debug-list-item-error' key={index}>{log.message}</li>
            : <li className='Debug-list-item-log' key={index}>{log.message}</li>
        })}
      </ul>
      <div className='Debug-buttons'>
        {paused
          ? <button onClick={debug.resume}>Resume</button> 
          : <button onClick={debug.pause} >Pause</button> }
        <button onClick={debug.clear} >Clear</button>
      </div>
      
    </div>
  </div>)
}

export default Debug