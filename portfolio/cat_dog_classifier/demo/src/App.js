import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera/Camera'
import Info from './components/Info/Info'
import Results from './components/Results/Results';
import Debug from './components/Debug/Debug';

const animal_class_map = ['Cat', 'Dog']
const cat_class_map = ['Abyssinian','American_Bobtail','American_Curl','American_Shorthair','American_Wirehair','Applehead_Siamese','Balinese','Bengal','Birman','Bombay','British_Shorthair','Burmese','Burmilla','Calico','Canadian_Hairless','Chartreux','Chausie','Chinchilla','Cornish_Rex','Cymric','Devon_Rex','Dilute_Calico','Dilute_Tortoiseshell','Domestic_Long_Hair','Domestic_Medium_Hair','Domestic_Short_Hair','Egyptian_Mau','Exotic_Shorthair','Havana','Himalayan','Japanese_Bobtail','Javanese','Korat','LaPerm','Maine_Coon','Manx','Munchkin','Nebelung','Norwegian_Forest_Cat','Ocicat','Oriental_Long_Hair','Oriental_Short_Hair','Oriental_Tabby','Persian','Pixiebob','Ragamuffin','Ragdoll','Russian_Blue','Scottish_Fold','Selkirk_Rex','Siamese','Siberian','Silver','Singapura','Snowshoe','Somali','Sphynx','Tabby','Tiger','Tonkinese','Torbie','Tortoiseshell','Turkish_Angora','Turkish_Van','Tuxedo','York_Chocolate']
const dog_class_map = ['Affenpinscher','Afghan_Hound','African_Hunting_Dog','Airedale','American_Staffordshire_Terrier','Appenzeller','Australian_Terrier','Basenji','Basset','Beagle','Bedlington_Terrier','Bernese_Mountain_Dog','Black-and-Tan_Coonhound','Blenheim_Spaniel','Bloodhound','Bluetick','Border_Collie','Border_Terrier','Borzoi','Boston_Bull','Bouvier_des_Flandres','Boxer','Brabancon_Griffon','Briard','Brittany_Spaniel','Bull_Mastiff','Bulldog','Cairn','Cardigan','Chesapeake_Bay_Retriever','Chihuahua','Chow','Clumber','Cocker_Spaniel','Collie','Curly-Coated_Retriever','Dandie_Dinmont','Dhole','Dingo','Doberman','English_Foxhound','English_Setter','English_Springer','EntleBucher','Eskimo_Dog','Flat-Coated_Retriever','German_Shepherd','German_Short-Haired_Pointer','Giant_Schnauzer','Golden_Retriever','Gordon_Setter','Great_Dane','Great_Pyrenees','Greater_Swiss_Mountain_Dog','Groenendael','Havanese','Ibizan_Hound','Irish_Setter','Irish_Terrier','Irish_Water_Spaniel','Irish_Wolfhound','Italian_Greyhound','Japanese_Chin','Japanese_Spaniel','Keeshond','Kelpie','Kerry_Blue_Terrier','Komondor','Kuvasz','Labrador_Retriever','Lakeland_Terrier','Leonberg','Lhasa','Malamute','Malinois','Maltese','Mexican_Hairless','Miniature_Pinscher','Miniature_Poodle','Miniature_Schnauzer','Newfoundland','Norfolk_Terrier','Norwegian_Elkhound','Norwich_Terrier','Old_English_Sheepdog','Otterhound','Papillon','Pekinese','Pembroke','Pitbull','Pomeranian','Pug_','Redbone','Rhodesian_Ridgeback','Rottweiler','Saint_Bernard','Saluki','Samoyed','Schipperke','Scotch_Terrier','Scottish_Deerhound','Sealyham_Terrier','Shetland_Sheepdog','Shiba_Inu','Shih-Tzu','Siberian_Husky','Silky_Terrier','Soft-Coated_Wheaten_Terrier','Staffordshire_Bullterrier','Standard_Poodle','Standard_Schnauzer','Sussex_Spaniel','Tibetan_Mastiff','Tibetan_Terrier','Toy_Poodle','Toy_Terrier','Vizsla','Walker_Hound','Weimaraner','Welsh_Springer_Spaniel','West_Highland_White_Terrier','Wheaten_Terrier','Whippet','Wire-Haired_Fox_Terrier','Yorkshire_Terrier']

const animalCertaintyThreshold = 0.7

class App extends React.Component {
  constructor(props) {
    super(props);
    this.modelURL = 'https://built-model-repository.s3.us-west-2.amazonaws.com/cat_dog_classifier/combined_model/model.json'
    this.state = {
      animalDetected: false,
      loading: true,
      animalClass: 'Cat',
      animalProb: 0,
      breedClass: 'an English Shorthair',
      breedProb: 0,
      model: undefined,
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
    const model = await tf.loadGraphModel(this.modelURL);
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
        const [dog_probs, cat_probs, animal_probs, animal_linear] = this.state.model.predict(tensor)
        const animalDetected = animal_linear.max().dataSync()[0] > animalCertaintyThreshold
        const animalClass = animal_class_map[animal_probs.argMax(-1).dataSync()[0]]
        const animalProb = Math.floor(animal_probs.max().dataSync()[0] * 100)
        
        let breedClass = 'unknown breed'
        let breedProb = 0
        if(animalClass === 'Cat') {
          breedClass = cat_class_map[cat_probs.argMax(-1).dataSync()[0]]
          breedProb =  Math.floor(cat_probs.max().dataSync()[0] * 100)
        } else if (animalClass === 'Dog') {
          breedClass = dog_class_map[dog_probs.argMax(-1).dataSync()[0]]
          breedProb =  Math.floor(dog_probs.max().dataSync()[0] * 100)
        }

        this.debug.log(`animal linear: ${animal_linear.max().dataSync()[0]}`)

        this.setState({
          animalDetected,
          animalClass,
          animalProb,
          breedClass,
          breedProb,
        })
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
