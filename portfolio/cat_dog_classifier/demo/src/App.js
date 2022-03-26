import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import Camera from './components/Camera'
import Info from './components/Info'

const animal_class_map = ['Cat', 'Dog']
const cat_class_map = ['Abyssinian','American_Bobtail','American_Curl','American_Shorthair','American_Wirehair','Applehead_Siamese','Balinese','Bengal','Birman','Bombay','British_Shorthair','Burmese','Burmilla','Calico','Canadian_Hairless','Chartreux','Chausie','Chinchilla','Cornish_Rex','Cymric','Devon_Rex','Dilute_Calico','Dilute_Tortoiseshell','Domestic_Long_Hair','Domestic_Medium_Hair','Domestic_Short_Hair','Egyptian_Mau','Exotic_Shorthair','Havana','Himalayan','Japanese_Bobtail','Javanese','Korat','LaPerm','Maine_Coon','Manx','Munchkin','Nebelung','Norwegian_Forest_Cat','Ocicat','Oriental_Long_Hair','Oriental_Short_Hair','Oriental_Tabby','Persian','Pixiebob','Ragamuffin','Ragdoll','Russian_Blue','Scottish_Fold','Selkirk_Rex','Siamese','Siberian','Silver','Singapura','Snowshoe','Somali','Sphynx','Tabby','Tiger','Tonkinese','Torbie','Tortoiseshell','Turkish_Angora','Turkish_Van','Tuxedo','York_Chocolate']
const dog_class_map = ['Affenpinscher','Afghan_Hound','African_Hunting_Dog','Airedale','American_Staffordshire_Terrier','Appenzeller','Australian_Terrier','Basenji','Basset','Beagle','Bedlington_Terrier','Bernese_Mountain_Dog','Black-and-Tan_Coonhound','Blenheim_Spaniel','Bloodhound','Bluetick','Border_Collie','Border_Terrier','Borzoi','Boston_Bull','Bouvier_des_Flandres','Boxer','Brabancon_Griffon','Briard','Brittany_Spaniel','Bull_Mastiff','Bulldog','Cairn','Cardigan','Chesapeake_Bay_Retriever','Chihuahua','Chow','Clumber','Cocker_Spaniel','Collie','Curly-Coated_Retriever','Dandie_Dinmont','Dhole','Dingo','Doberman','English_Foxhound','English_Setter','English_Springer','EntleBucher','Eskimo_Dog','Flat-Coated_Retriever','German_Shepherd','German_Short-Haired_Pointer','Giant_Schnauzer','Golden_Retriever','Gordon_Setter','Great_Dane','Great_Pyrenees','Greater_Swiss_Mountain_Dog','Groenendael','Havanese','Ibizan_Hound','Irish_Setter','Irish_Terrier','Irish_Water_Spaniel','Irish_Wolfhound','Italian_Greyhound','Japanese_Chin','Japanese_Spaniel','Keeshond','Kelpie','Kerry_Blue_Terrier','Komondor','Kuvasz','Labrador_Retriever','Lakeland_Terrier','Leonberg','Lhasa','Malamute','Malinois','Maltese','Mexican_Hairless','Miniature_Pinscher','Miniature_Poodle','Miniature_Schnauzer','Newfoundland','Norfolk_Terrier','Norwegian_Elkhound','Norwich_Terrier','Old_English_Sheepdog','Otterhound','Papillon','Pekinese','Pembroke','Pitbull','Pomeranian','Pug_','Redbone','Rhodesian_Ridgeback','Rottweiler','Saint_Bernard','Saluki','Samoyed','Schipperke','Scotch_Terrier','Scottish_Deerhound','Sealyham_Terrier','Shetland_Sheepdog','Shiba_Inu','Shih-Tzu','Siberian_Husky','Silky_Terrier','Soft-Coated_Wheaten_Terrier','Staffordshire_Bullterrier','Standard_Poodle','Standard_Schnauzer','Sussex_Spaniel','Tibetan_Mastiff','Tibetan_Terrier','Toy_Poodle','Toy_Terrier','Vizsla','Walker_Hound','Weimaraner','Welsh_Springer_Spaniel','West_Highland_White_Terrier','Wheaten_Terrier','Whippet','Wire-Haired_Fox_Terrier','Yorkshire_Terrier']

// const detectionBias = [-0.1, -0.5]
const animalCertaintyThreshold = 0.6

const isVowel = (char) => {
  char.toUpperCase()
  return char === "A" || char === "E" || char === "I" || char === "O" || char === "U";
}

const formatClassName = (className) => {
  return `${isVowel(className[0]) ? 'an' : 'a'} ${className.replace('_', ' ')}`
}

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

  componentDidMount = async () => {
    const model = await tf.loadGraphModel('https://built-model-repository.s3.us-west-2.amazonaws.com/cat_dog_classifier/combined_model/model.json');
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
          const [dog_probs, cat_probs, animal_logits] = this.state.model.predict(tensor)
          const animal_linear = tf.sigmoid(animal_logits)
          const animal_softmax = tf.softmax(animal_logits)
          const animalClass = animal_class_map[animal_softmax.argMax(-1).dataSync()[0]]
          let breedClass = 'unknown breed'
          let breedProb = 0

          // console.log('animal linear:', animal_linear.max().dataSync()[0])
          
          if(animalClass === 'Cat') {
            breedClass = cat_class_map[cat_probs.argMax(-1).dataSync()[0]]
            breedProb =  cat_probs.max().dataSync()[0]
          } else {
            breedClass = dog_class_map[dog_probs.argMax(-1).dataSync()[0]]
            breedProb =  dog_probs.max().dataSync()[0]
          }

          this.setState({
            animalDetected: animal_linear.max().dataSync()[0] > animalCertaintyThreshold,
            animalClass,
            animalProb: Math.floor(animal_softmax.max().dataSync()[0] * 100),
            breedClass,
            breedProb: Math.floor(breedProb * 100),
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
    <p>Maybe {formatClassName(this.state.breedClass)}? ({this.state.breedProb}%)</p>
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
