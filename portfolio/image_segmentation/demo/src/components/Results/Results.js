import React from "react";
import './Results.css'

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
    this.setState({context, viewportWidth}, this.drawMasks)
  }

  componentDidUpdate() {
    this.drawMasks()
  }

  renderText = () => {
    if (this.props.state.loading) {
      return <div className="Results-text"><div className="Results-loader"></div><p>Loading model...</p></div>
    } else {
      return <p>Detected classes in scene:</p>
    }
  } 

  drawMasks = async () => {
    if (this.state.context
      && this.state.viewportWidth
      && this.props.state
      && this.props.state.masks) {
      // console.log(this.props.state.classes)
      this.state.context.clearRect(0, 0, this.state.viewportWidth, this.state.viewportWidth)
      createImageBitmap(this.props.state.masks).then(
        (img) => this.state.context.drawImage(img, 0, 0, this.state.viewportWidth, this.state.viewportWidth)
      )
    }
  }

  onSlideChange = (e) => {
    console.log(Math.log10(parseFloat(e.target.value)))
    this.props.onSlideChange(Math.log10(parseFloat(e.target.value)))
  }
  // value={Math.pow(10, this.props.state.softmax_threshold)}

  render() {
    return (
    <div className='Results'>
      <canvas className="Results-masks-container" height={224} width={224}></canvas>
      <div className="Results-slider">
      <div className="Results-slider-value">Softmax threshold for pixel classification: {this.props.state.softmax_threshold.toFixed(2)}</div>
      <input className="Results-slider-range" type="range" min="7" max="10" step="0.001" onChange={this.onSlideChange} value={Math.pow(10, this.props.state.softmax_threshold)} />
      </div>
      <div className="Results-detected">
        {this.renderText()}
      </div>
      <div className="Results-detected">
        {this.props.state.classes.map((_class, i) => {
          return <div className="Results-span" style={{backgroundColor: _class.color}} key={`list${i}`}>{_class.name}</div>
        })}
      </div>
    </div>
    )
  }
}

export default Results
