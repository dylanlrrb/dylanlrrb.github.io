import React from "react";
import './Results.css'

// const dpr = window.devicePixelRatio || 1
const dpr = 1

class Results extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      context: undefined,
      viewportWidth: undefined,
    }
  }

  componentDidMount() {
    const canvas = document.querySelector('.Results-masks-container')
    const context = canvas.getContext("2d")
    const viewportWidth = Math.floor(document.querySelector('.App').getBoundingClientRect().width)
    canvas.setAttribute('height', viewportWidth)
    canvas.setAttribute('width', viewportWidth)
    this.setState({context, viewportWidth}, this.drawKeypoints)
  }

  componentDidUpdate() {
    this.drawKeypoints()
  }

  drawKeypoints = () => {
    if (this.state.context && this.state.viewportWidth) {
      
    }
  }


  // this.props.debug.log(`${}`)

  render() {
    return (
      <div className='Results'>
        <canvas className="Results-masks-container" height={256} width={256}></canvas>
      </div>
    )
  }
    
}

export default Results
