import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
// import AdvancedCamera from './components/AdvancedCamera/AdvancedCamera'
import Results from './components/Results/Results';
import Info from './components/Info/Info'
import Debug from './components/Debug/Debug';
// https://docs.opencv.org/4.x/de/d06/tutorial_js_basic_ops.html
// export NODE_OPTIONS="--max-old-space-size=8192"
var cv = require('opencv.js')

const zip = (...arr) => {
  const zipped = [];
  arr.forEach((element, ind) => {
     element.forEach((el, index) => {
        if(!zipped[index]){
           zipped[index] = [];
        };
        if(!zipped[index][ind]){
           zipped[index][ind] = [];
        }
        zipped[index][ind] = el || '';
     })
  });
  return zipped;
};

class App extends React.Component {
  constructor(props) {
    super(props);
    this.model_dim = 224
    this.expand_factor = 1.3

    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/mobilenet_backbone_2/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/mobilenet_backbone_3/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/mobilenet_backbone_full/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/mobilenet_backbone_full_30pct/model.json'
    // this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/mobilenet_backbone_full_30pct_all_points/model.json' // baseline
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/facial_keypoints/mobilenet_backbone_full_non_trainable/model.json' // actually does pretty good at not jumping around, not as tight tho

    this.state = {
      loading: true,
      model: undefined,
      logs: [],
      paused: false,
      keypoints: [],
      bboxes: [],
      testCanvas: undefined,
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

    // https://github.com/huningxin/opencv/blob/master/doc/js_tutorials/js_assets/js_face_detection.html
    // https://github.com/huningxin/opencv/blob/master/doc/js_tutorials/js_assets/utils.js
    let faceCascadeFile = 'https://raw.githubusercontent.com/huningxin/opencv/master/data/haarcascades_cuda/haarcascade_frontalface_default.xml';
    this.createFileFromUrl('haarcascade_frontalface_default.xml', faceCascadeFile, () => {
      this.faceClassifier = new cv.CascadeClassifier();
      this.faceClassifier.load('haarcascade_frontalface_default.xml');
    })

    const model = await tf.loadGraphModel(this.modelURL);
    let testCanvas = document.querySelector('#testCanvas')
    this.setState({
      loading: false,
      model,
      testCanvas,
    })
  }

  createFileFromUrl = (path, url, callback) => {
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = (ev) => {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                callback();
            } else {
                console.log('Failed to load ' + url + ' status: ' + request.status);
            }
        }
    };
    request.send();
  };

  findFaces = (imageData, viewportWidth) => {
    const bboxes = []
    let src = cv.matFromImageData(imageData)
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    let faces = new cv.RectVector();
    let minsize = new cv.Size(50, 50);
    let maxsize = new cv.Size(viewportWidth, viewportWidth);
    // https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html
    this.faceClassifier.detectMultiScale(gray, faces, 1.2, 3, 0, minsize, maxsize);
    for (let i = 0; i < faces.size(); ++i) {
        const expand_pixels = Math.trunc(faces.get(i).width * this.expand_factor) - faces.get(i).width
        const bbox = {
          x: Math.max(faces.get(i).x - Math.trunc(expand_pixels / 2), 0),
          y: Math.max(faces.get(i).y - Math.trunc(expand_pixels / 2), 0),
          height: faces.get(i).height,
          width: faces.get(i).width,
        }
        bbox.width = bbox.width + expand_pixels + bbox.x > viewportWidth ? (viewportWidth - bbox.x) : (bbox.width + expand_pixels)
        bbox.height = bbox.height + expand_pixels + bbox.y > viewportWidth ? (viewportWidth - bbox.y) : (bbox.height + expand_pixels)
        bboxes.push(bbox)
    }
    src.delete(); gray.delete(); faces.delete();
    return bboxes
  }

  cropFaces = (image, bboxes) => {
    // return array of face crops to make predictions on
    let faceCrops = []
    bboxes.forEach((bbox) => {
      let faceCrop = tf.slice3d(image, [bbox.x, bbox.y, 0], [bbox.width, bbox.height, 3])
      // faceCrop = tf.image.resizeBilinear(faceCrop, [Math.trunc(this.model_dim / 2),Math.trunc(this.model_dim / 2)])
      faceCrop = tf.image.resizeBilinear(faceCrop, [this.model_dim,this.model_dim])
      faceCrops.push(faceCrop)
    })
    return faceCrops
  }

  getKeypoints = (faceCrops, bboxes, viewportWidth) => {
    return zip(faceCrops, bboxes).map(([faceCrop, bbox]) => {
      let keypoints = this.state.model.predict(tf.expandDims(faceCrop, 0))
      keypoints = tf.squeeze(keypoints)
      keypoints = tf.add(keypoints, 1)
      keypoints = keypoints.reshape([-1, 2])
      keypoints = tf.mul(keypoints, [bbox.width / 2, bbox.height / 2])
      keypoints = tf.add(keypoints, [bbox.x, bbox.y])
      return keypoints.arraySync()
    })
  }



  predict = (imageTensor, imageData, viewportWidth) => {
    if (this.state.model && viewportWidth && this.faceClassifier) {
      let [bboxes, keypoints] = tf.tidy(() => {
        const bboxes = this.findFaces(imageData, viewportWidth)
        if (bboxes.length > 0){
          const faceCrops = this.cropFaces(imageTensor, bboxes)
          const keypoints = this.getKeypoints(faceCrops, bboxes, viewportWidth)
          return [bboxes, keypoints]
        }
        return [[], []]
      })
      this.setState({bboxes, keypoints})
    }
  }



  render() {
    return (
      <div className="App">
        <Camera predict={this.predict} debug={this.debug} />
        <Results state={this.state} />
        <canvas id="testCanvas"></canvas>
        <Info />
        <Debug debug={this.debug} logs={this.state.logs} paused={this.state.paused} />
      </div>
    )
  }
}

export default App;
