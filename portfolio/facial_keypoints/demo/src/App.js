import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
import AdvancedCamera from './components/AdvancedCamera/AdvancedCamera'
import Results from './components/Results/Results';
import Info from './components/Info/Info'
import Debug from './components/Debug/Debug';


const range = (start, end, skip=1) => {
  return [...Array(end).keys()].filter(i => i % skip === 0 && i >= start)
}


class App extends React.Component {
  constructor(props) {
    super(props);
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/facial_keypoints/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/facial_keypoints_last_conv/model.json'

    this.state = {
      loading: true,
      model: undefined,
      logs: [],
      paused: false,
      keypoints: [],
    }

    this.debug = {
      log: ((message) => {
        const {logs} = this.state
        !this.state.paused && logs.push({type: 'log', message})
        this.state.logs.length > 100 && logs.shift()
        this.setState({logs})
      }),
      error: ((message) => {
        const {logs} = this.state
        !this.state.paused && logs.push({type: 'error', message})
        this.state.logs.length > 100 && logs.shift()
        this.setState({logs})
      }),
      pause: (() => {this.setState({paused: true})}),
      resume: (() => {this.setState({paused: false})}),
      clear: (() => {this.setState({logs: []})}),
    }
  }

  componentDidCatch(error, errorInfo) {
    this.debug.error(error, errorInfo);
  }

  componentDidMount = async () => {
    const model = await tf.loadGraphModel(this.modelURL);
    this.setState({
      loading: false,
      model
    })
  }

  predict = (tensor) => {
    if (this.state.model) {
      let keypoints = tf.tidy(() => {
        const rgb = tf.tensor1d([0.2989, 0.587, 0.114])
        let input = tf.div(tensor, tf.scalar(255))
        input = tf.sum(image.mul(rgb), 2)
        input = tf.image.resizeNearestNeighbor(input, [256,256]).toFloat()
        input = tf.expandDims(input, 0)
        let keypoints = tf.squeeze(this.state.model.predict(input))
        keypoints = tf.add(keypoints, tf.scalar(1))
        keypoints = keypoints.reshape([-1, 2])
        return tf.mul(keypoints, tf.tensor1d([256 / 2, 256 / 2]))
      })
      console.log(keypoints)
      this.setState({keypoints})
    }
  }



  render() {
    return (
      <div className="App">
        <Camera predict={this.predict} />
        <Results state={this.state} />
        <Info />
        <Debug debug={this.debug} logs={this.state.logs} paused={this.state.paused} />
      </div>
    )
  }
}

export default App;
