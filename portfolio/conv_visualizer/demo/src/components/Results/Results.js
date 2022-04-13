import React from "react";
import './Results.css'
import * as tf from "@tensorflow/tfjs"


const randomBetween = (min, max) => min + Math.floor(Math.random() * (max - min + 1));

const randomColor = () => {
  const random = randomBetween(0, 255)
  return `rgba(${random},${random},${random},0.3)`;
}


class Results extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      selectedFilter: 0,
      filterCount: 60,
      filters: [],
      filtersContainer: undefined,
    }
  }

  componentDidMount() {
    this.setState({
      filters: Array.from(document.querySelectorAll('canvas.filter')),
      filtersContainer: document.querySelector('.Results-filters')})
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevProps.state.output !== this.props.state.output) {
      this.state.filters.forEach((canvas, i) => {
        const outputs = this.props.state.output.sort((a, b) => b.size - a.size)
        let tensor = outputs[this.state.selectedFilter]
        if (tensor) {
          tensor = tf.squeeze(tensor)
          tensor = tf.slice3d(tensor, [0,0,i], [-1, -1, 1])
          const scale = tensor.max().sub(tensor.min())
          tensor = tensor.sub(tensor.min()).div(scale)
          tf.browser.toPixels(tensor, canvas)
        }
      })
    }
  }

  setSelectedFilter = (filterNum) => {
    return () => {
      this.state.filtersContainer.scrollTop = 0
      this.setState({selectedFilter: filterNum})
    }
  }

  renderFilters = () => {
    return (new Array(this.state.filterCount)).fill(null).map((_, i) => {
      return <canvas className="filter" key={i} style={{backgroundColor: randomColor(), height: 'calc(var(--vw, 1vw) * 33.33)', width: 'calc(var(--vw, 1vw) * 33.33)'}}></canvas>
    })
  }


  render() {
    return (
      <div className='Results'>
        <div className="Results-filters">
          {this.renderFilters()}
        </div>
        <div className="Results-selector">
          <p>Convolutional Layers:</p>
          <div className="Results-pills">
            <span className={this.state.selectedFilter === 0 ? 'highlight' : ''} onClick={this.setSelectedFilter(0)}>Conv_1</span>
            <span className={this.state.selectedFilter === 1 ? 'highlight' : ''} onClick={this.setSelectedFilter(1)}>Conv_2</span>
            <span className={this.state.selectedFilter === 2 ? 'highlight' : ''} onClick={this.setSelectedFilter(2)}>Conv_3</span>
            <span className={this.state.selectedFilter === 3 ? 'highlight' : ''} onClick={this.setSelectedFilter(3)}>Conv_4</span>
            <span className={this.state.selectedFilter === 4 ? 'highlight' : ''} onClick={this.setSelectedFilter(4)}>Conv_5</span>
            <span className={this.state.selectedFilter === 5 ? 'highlight' : ''} onClick={this.setSelectedFilter(5)}>Conv_6</span>
            <span className={this.state.selectedFilter === 6 ? 'highlight' : ''} onClick={this.setSelectedFilter(6)}>Conv_7</span>
            <span className={this.state.selectedFilter === 7 ? 'highlight' : ''} onClick={this.setSelectedFilter(7)}>Conv_8</span>
            <span className={this.state.selectedFilter === 8 ? 'highlight' : ''} onClick={this.setSelectedFilter(8)}>Conv_9</span>
            <span className={this.state.selectedFilter === 9 ? 'highlight' : ''} onClick={this.setSelectedFilter(9)}>Conv_10</span>
            <span className={this.state.selectedFilter === 10 ? 'highlight' : ''} onClick={this.setSelectedFilter(10)}>Conv_11</span>
            <span className={this.state.selectedFilter === 11 ? 'highlight' : ''} onClick={this.setSelectedFilter(11)}>Conv_12</span>
          </div>
          {/* <p>Fully Connected Layers:</p>
          <div className="Results-pills">
            <span>first layer</span>
            <span>second layer</span>
            <span>third layer</span>
            <span>fourth layer</span>
            <span>fith layer</span>
            <span>sixth layer</span>
          </div> */}
        </div>
       
     </div>
    )
  }
    
}

export default Results
