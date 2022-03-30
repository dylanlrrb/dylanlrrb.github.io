import React from "react";
import './Results.css'

const isVowel = (char) => {
  char.toUpperCase()
  return char === "A" || char === "E" || char === "I" || char === "O" || char === "U";
}

const formatClassName = (className) => {
  return `${isVowel(className[0]) ? 'an' : 'a'} ${className.replace('_', ' ')}`
}


class Results extends React.Component {
  
  renderIcon = () => {
    if (!this.props.state.animalDetected) {
      return <div className='Results-loader'></div>
    }
    if (this.props.state.animalClass === 'Cat') {
      return <div className='Results-icon'>🐱</div>
    }
    return <div className='Results-icon'>🐶</div>
  }

  renderText = () => {
    if (this.props.state.loading) {
      return <div><p>Loading model...</p></div>
    }
    if (!this.props.state.animalDetected) {
      return <div><p>No animal detected in frame.</p>
        <p>Point your camera at a pet!</p></div>
    }
    return <div><p>I think this is a {this.props.state.animalClass}! ({this.props.state.animalProb}%)</p>
      <p>Maybe {formatClassName(this.props.state.breedClass)}? ({this.props.state.breedProb}%)</p>
      </div>

     
  }

  render() {
    return (
      <div className='Results'>
       <div>
         {this.renderIcon()}
       </div>
       {this.renderText()}
     </div>
    )
  }
    
}

export default Results
