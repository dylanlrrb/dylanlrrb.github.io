import React from "react";
import './Results.css'

// const dpr = window.devicePixelRatio || 1
const dpr = 1

class Results extends React.Component {

  constructor(props) {
    super(props);
    this.maxZoom = 5
    this.state = {
      viewportWidth: 0,
      slideValue: 50,
      originalContext: undefined,
      enhancedContext: undefined,
      // coordinates
      isPanning: false,
      x: 0,
      y: 0,
      isPinching: false,
      previousTouch: undefined,
      pinchDist: undefined,
      dimension: undefined,
    }
  }

  componentDidMount() {
    // draw images to appropriate canvases
    const  viewportWidth = Math.floor(document.querySelector('.App').getBoundingClientRect().width)
    
    let canvasOriginal = document.querySelector('#Results-canvas-original')
    canvasOriginal.setAttribute('height', viewportWidth)
    canvasOriginal.setAttribute('width', viewportWidth)
    const originalContext = canvasOriginal.getContext('2d');
    originalContext.scale(dpr, dpr)

    let canvasEnhanced = document.querySelector('#Results-canvas-enhanced')
    canvasEnhanced.setAttribute('height', viewportWidth)
    canvasEnhanced.setAttribute('width', viewportWidth)
    const enhancedContext = canvasEnhanced.getContext('2d');
    enhancedContext.scale(dpr, dpr)

    if (this.props.originalImg && this.props.enhancedImg) {
      createImageBitmap(this.props.originalImg).then(
        (img) => originalContext.drawImage(img, 0, 0)
      )
      createImageBitmap(this.props.enhancedImg).then(
        (img) => enhancedContext.drawImage(img, 0, 0)
      )
    }

    this.setState({originalContext, enhancedContext, viewportWidth, dimension: viewportWidth})
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevState.x !== this.state.x || prevState.y !== this.state.y || prevState.dimension !== this.state.dimension) {
      if (this.props.originalImg &&
          this.props.enhancedImg &&
          this.state.originalContext &&
          this.state.enhancedContext) {
        createImageBitmap(this.props.originalImg).then(
          (img) => this.state.originalContext.drawImage(img, this.state.x, this.state.y, this.state.dimension / (dpr), this.state.dimension / (dpr))
        )
        createImageBitmap(this.props.enhancedImg).then(
          (img) => this.state.enhancedContext.drawImage(img, this.state.x, this.state.y, this.state.dimension / (dpr), this.state.dimension / (dpr))
        )
      }
    }
  }

  onSlideChange = (e) => {
    this.setState({slideValue: e.target.value})
  }

  onMouseDown = () => {
    this.setState({isPanning: true})
  }

  onMouseUp = () => {
    this.setState({isPanning: false})
  }

  constrainX = (x) => {
    if (x < (this.state.dimension - this.state.viewportWidth) * -1) {
      return (this.state.dimension - this.state.viewportWidth) * -1
    } else if (x > 0) {
      return 0
    } else {
      return x
    }
  }

  constrainY = (y) => {
    if (y < (this.state.dimension - this.state.viewportWidth) * -1) {
      return (this.state.dimension - this.state.viewportWidth) * -1
    } else if (y > 0) {
      return 0
    } else {
      return y
    }
  }

  constrainZoom = (z) => {
    if (z < this.state.viewportWidth) {
      return this.state.viewportWidth
    } else {
      return z
    }
  }

  onMouseMove = (e) => {
    if (this.state.isPanning) {
      this.setState({x: this.constrainX(this.state.x + e.movementX), y: this.constrainY(this.state.y + e.movementY)})
    }
  }

  onMouseScroll = (e) => {
    this.setState({dimension: this.constrainZoom(this.state.dimension + e.deltaY), x: this.constrainX(this.state.x - (e.deltaY/2)), y: this.constrainY(this.state.y - (e.deltaY/2))})
  }

  onTouchStart = (e) => {
    if (e.touches.length === 2) {
        this.setState({isPinching: true, pinchDist: undefined})
    } else {
      this.setState({isPanning: true, previousTouch: undefined})
    }

   
  }

  onTouchMove = (e) => {
    if (this.state.isPinching) {
      const pinchDist = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY);
      let dimension = this.state.dimension
      let focusPointX =  (e.touches[0].pageX + e.touches[1].pageX)/2
      let focusPointY =  (e.touches[0].pageY + e.touches[1].pageY)/2
      let x = this.state.x
      let y = this.state.y
      if (this.state.pinchDist) {
        const deltaPinch = pinchDist - this.state.pinchDist
        dimension = this.constrainZoom(dimension + deltaPinch)
        x = this.constrainX(x - (focusPointX / (deltaPinch * 2)))
        y = this.constrainY(y - (focusPointY / (deltaPinch * 2)))
      }
      this.setState({pinchDist, dimension, x, y})
    }

    if (this.state.isPanning) {
      const touch = e.touches[0];
      e.movementX = 0
      e.movementY = 0
      if (this.state.previousTouch) {
        e.movementX = touch.pageX - this.state.previousTouch.pageX;
        e.movementY = touch.pageY - this.state.previousTouch.pageY;
      }
      this.setState({previousTouch: touch, x: this.constrainX(this.state.x + e.movementX), y: this.constrainY(this.state.y + e.movementY)})
    }
  }

  onTouchEnd = () => {
    this.setState({isPinching: false, isPanning: false})
  }

  // this.props.debug.log(`${}`)

  render() {
    return (
      <div className='Results'>
        <div
          className="Results-controls"
          onMouseDown={this.onMouseDown}
          onMouseUp={this.onMouseUp}
          onMouseMove={this.onMouseMove}
          onWheel={this.onMouseScroll}
          onTouchStart={this.onTouchStart}
          onTouchMove={this.onTouchMove}
          onTouchEnd={this.onTouchEnd}
        ></div>
        
        <div className="Results-split">
          <div className="Results-original">
            <canvas id="Results-canvas-original"></canvas>
          </div>
          <div className="Results-enhanced" style={{width: `${this.state.slideValue}%`}}>
            <canvas id="Results-canvas-enhanced"></canvas>
          </div>
          <input className="Results-slider" type="range" min="1" max="100" step="1" value={this.state.slideValue} onChange={this.onSlideChange}/>
        </div>

        <div className='Results-retake-button'>
          <button onClick={this.props.retake}><h2>Retake</h2></button> 
        </div>

      </div>
    )
  }
    
}

export default Results
