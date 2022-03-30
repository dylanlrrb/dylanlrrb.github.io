import React from "react";
import './Results.css'

class Results extends React.Component {

  renderText = () => {
    if (this.props.state.loading) {
      return <div className="Results-text"><div className="Results-loader"></div><p>Loading model...</p></div>
    } else {
      return <p>Detected classes in scene:</p>
    }
  } 

  render() {
    return (
    <div className='Results'>
      <div className="Results-boxes-container">
        {this.props.state.predictions.map((prediction, i) => {
          {/* if (prediction.score < 0.4) {return} */}
          const [left, top, width, height] = prediction.bbox
          return <div key={`box${i}`}>
            <div
              className="Results-box-highlight"
              style={{'marginLeft': left-2, 'marginTop': top-2, 'width': width, 'height': height}}
            >
            </div>
            <p
              className="Results-box-label"
              style={{'marginLeft': left, 'marginTop': top,}}
            >
              {prediction.class} {Math.round(prediction.score * 100)}%
            </p>
          </div>
        })}
      </div>
      <div className="Results-detected">
        {this.renderText()}
      </div>
      <div className="Results-detected">
        {this.props.state.predictions.map((prediction, i) => {
          return <div className="Results-span" key={`list${i}`}>{prediction.class}</div>
        })}
      </div>
    </div>
    )
  }
}

export default Results
