import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
import Info from './components/Info/Info'
import Results from './components/Results/Results';
import Debug from './components/Debug/Debug';
import * as mobilenet from '@tensorflow-models/mobilenet'

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: true,
      model: undefined,
      output: [],
      logs: [],
      paused: false,
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
    const model = await tf.loadGraphModel('https://built-model-repository.s3.us-west-2.amazonaws.com/conv_visualizer/model.json');
    this.setState({
      loading: false,
      model
    })
  }

  predict = (tensor) => {
    tf.tidy(() => {
      if (this.state.model) {
        tensor = tf.image.resizeNearestNeighbor(tensor, [224,224]).toFloat()
        tensor = tf.expandDims(tensor, 0)
        const output = this.state.model.predict(tensor)
        this.setState({output})
      }
    })
    this.debug.log(`num Tensors: ${tf.memory().numTensors}`)
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
