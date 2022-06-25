import React from 'react';
import * as tf from "@tensorflow/tfjs"
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import './App.css';
import Camera from './components/Camera/Camera'
import Info from './components/Info/Info'
import Results from './components/Results/Results';
import Debug from './components/Debug/Debug';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      model: undefined,
      loading: true,
      predictions: [],
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
    const model = await cocoSsd.load()
    this.setState({
      loading: false,
      model
    })
  }

  predict = async (tensor) => {
    if (this.state.model) {
      const predictions = await this.state.model.detect(tensor)
      // console.log(predictions)
      this.setState({predictions})
    }
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
