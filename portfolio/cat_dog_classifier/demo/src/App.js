import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera'
import Info from './components/Info'
import { math } from '@tensorflow/tfjs';

const animalMap = ['Cat', 'Dog']
const animalCertaintyThreshold = 0.3
const catBreedMap = []
const dogBreedMap = []

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      animalDetected: false,
      loading: true,
      animalClass: 'Cat',
      animalProb: 0,
      breedClass: 'an English Shorthair',
      breedProb: 0,

      animalModel: undefined,
      catModel: undefined,
      dogModel: undefined,
      model: undefined,
    }
  }

  // componentDidMount = async () => {
  //   const featureDetector = await tf.loadLayersModel('http://model-server.domain/download/model.json');
  //   const animalClassifier = await tf.loadLayersModel('http://model-server.domain/download/model.json');
  //   const catBreedClassifier = await tf.loadLayersModel('http://model-server.domain/download/model.json');
  //   const dogBreedClassifier = await tf.loadLayersModel('http://model-server.domain/download/model.json');

  //   const animalModel = tf.sequential({layers: [featureDetector, animalClassifier]})
  //   const catModel = tf.Sequential({layers: [featureDetector, catBreedClassifier]})
  //   const dogModel = tf.Sequential({layers: [featureDetector, dogBreedClassifier]})

  //   this.setState({
  //     loading: false,
  //     animalModel,
  //     catModel,
  //     dogModel,
  //   })
  // }

  // predict = (tensor) => {
    // console.log(tensor)
    // const {animalModel, catModel, dogModel,} = this.state;
    // if (animalModel && catModel && dogModel) {
    //   const animal = animalModel.predict(tensor)
    //   const breed = undefined
    //   if (animal == 0) {
    //     breed = catBreedMap[catModel.predict(tensor)]
    //   } else {
    //     breed = dogBreedMap[dogModel.predict(tensor)]
    //   }
    //   this.setState({
    //     animalClass: animalMap[animal],
    //     animalProb: '??',
    //     breedClass: breed,
    //     breedProb: '??',
    //   })
    // }
  // }

  componentDidMount = async () => {
    const model = await tf.loadGraphModel('https://built-model-repository.s3.us-west-2.amazonaws.com/cat_dog_classifier/tfjs_model/model.json');
    // const model = await tf.loadLayersModel('https://built-model-repository.s3.us-west-2.amazonaws.com/cat_dog_classifier/test_tfjs_model/model.json');
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
          const logits = this.state.model.predict(tensor)
          const pred = tf.softmax(logits)
          const linear = tf.sigmoid(logits)
          // console.log(animalMap[pred.argMax(-1).dataSync()[0]], `${Math.floor(pred.max().dataSync()[0] * 100)}%`)
          linear.print()
          this.setState({
            animalDetected: linear.max().dataSync()[0] > animalCertaintyThreshold,
            animalClass: animalMap[pred.argMax(-1).dataSync()[0]],
            animalProb: Math.floor(pred.max().dataSync()[0] * 100),
            // breedClass: 'unknown breed',
            // breedProb: 0,
          })
        }
      })
    }

  renderIcon = () => {
    if (!this.state.animalDetected) {
      return <div className='App-results-loader'></div>
    }
    if (this.state.animalClass === 'Cat') {
      return <div className='App-results-icon'>🐱</div>
    }
    return <div className='App-results-icon'>🐶</div>
  }

  renderText = () => {
    if (this.state.loading) {
      return <div><p>Loading model...</p></div>
    }
    if (!this.state.animalDetected) {
      return <div><p>No animal detected in frame.</p>
      <p>Point your camera at a pet!</p></div>
    }
    return <div><p>I think this is a {this.state.animalClass}! ({this.state.animalProb}%)</p>
    {/* <p>Likely {this.state.breedClass} ({this.state.breedProb}%)</p> */}
    </div>

     
  }

  render() {
    return (
      <div className="App">
        <Camera predict={this.predict} />
        <Info />
        <div className='App-results'>
          <div>
            {this.renderIcon()}
          </div>
          {this.renderText()}
        </div>
      </div>
    )
  }
}

export default App;
