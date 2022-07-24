import React from "react";
import './Results.css'

// const dpr = window.devicePixelRatio || 1
// const dpr = 1

class Results extends React.Component {

  constructor(props) {
    super(props);
    this.model_dim = 256
    this.state = {
      showBboxes: false,
      context: undefined,
      viewportWidth: undefined,
    }
  }

  componentDidMount() {
    const canvas = document.querySelector('.Results-points-container')
    const context = canvas.getContext("2d")
    const viewportWidth = Math.floor(document.querySelector('.App').getBoundingClientRect().width)
    canvas.setAttribute('height', viewportWidth)
    canvas.setAttribute('width', viewportWidth)
    this.setState({context, viewportWidth}, this.draw)
  }

  componentDidUpdate() {
    this.draw()
  }

  draw = () => {
    this.state.context.clearRect(0, 0, this.state.viewportWidth, this.state.viewportWidth);
    if (this.state.showBboxes) {
      this.drawBBoxes()
    }
    this.drawKeypoints()
  }

  drawBBoxes = () => {
    if (this.state.context
      && this.state.viewportWidth
      && this.props.state.bboxes.length > 0) {
        // console.log(this.props.state.bboxes)

        const ctx = this.state.context
        
        this.props.state.bboxes.forEach((bbox) => {
          ctx.beginPath();
          ctx.lineWidth = "1";
          ctx.strokeStyle = "red";
          ctx.rect(bbox.x, bbox.y, bbox.width, bbox.height);
          ctx.stroke();
        })

      }
  }

  drawKeypoints = () => {
    if (this.state.context
      && this.state.viewportWidth
      && this.props.state.keypoints.length > 0) {
      // console.log(this.props.state.keypoints)
      const ctx = this.state.context

      this.props.state.keypoints.forEach((keypoint_set) => {
        keypoint_set.forEach((point) => {
          ctx.beginPath();
          ctx.fillStyle = "red";
          ctx.rect(point[0]-2, point[1]-2, 4, 4);
          ctx.fill();
        })
       
      })
      
    }
  }


  toggleStateProp = (prop) => {
    const state = {}
    state[prop] = !this.state[prop]
    this.setState(state)
  }

  render() {
    return (
      <div className='Results'>
        <canvas className="Results-points-container" height={this.model_dim} width={this.model_dim}></canvas>
        <label className="container">Show Face Bounding Boxes
          <input type="checkbox" onChange={() => this.toggleStateProp('showBboxes')} checked={this.state.showBboxes} />
          <span className="checkmark"></span>
        </label>
      </div>
    )
  }
    
}

export default Results
