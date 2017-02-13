var final_transcript = '';
var placeArray = [];
var clickArray = [];
var waypts = [];
var recognizing = false;
var ignore_onend;
var directionsService;
var directionsDisplay;
var geocoder;
var map; 
var markers = [];
var bounds;
var ignore_removeMarkers = false;
var fillerWords = ['', 'i', 'want', 'wanna','wish', 'would', 'like', 'to', 'two', 'go', 'visit', 'travel', 'and', 'in', 
'inn', 'plan', 'a', 'trip', 'between', 'the', '2','than', 'then', 'let\'s', 'lets', 'from', 'directions', 'what',
'how', 'get', 'do', 'since', 'home', 'here', 'when', 'goes', 'strip', 'that', 'anna', 'it', 'take', 'me'];
var prefixes = ['pre', 'el', 'la', 'las', 'los', 'san', 'north', 'south', 'east', 'west', 'new', 'mount', 'puerto'];
var postfixes = ['post', 'river', 'city', 'beach', 'island', 'cove', 'spring', 'springs', 'falls', 'valley', 'national', 'provincial', 'park', 'forest'];


//This function is intended to initialize  and display the map inside the div with the map id
function initMap() {
  directionsService = new google.maps.DirectionsService;
  directionsDisplay = new google.maps.DirectionsRenderer({suppressMarkers: true});
  map = new google.maps.Map(document.getElementById('map'), {   
  zoom: 2, //make a new map by specifying the element to put it in and passing an object full of the map's properties
  center: {lat: 28,lng: 2},
 });
 directionsDisplay.setMap(map);
 console.log("map created");
 
 google.maps.event.addListener(map, "click", addMarker)
}

//this function is supposed to take the directionsService and directionsDisplay objects and create 
function calculateAndDisplayRoute(directionsService, directionsDisplay) { //a roundtrip route out of the locations in the waypts array
  changeAlertBanner("#ff9800", "Calculating...");
  var places = {
   origin: placeArray[0],
   destination: placeArray[0], //the first location in the placesArray is used as the start and end location
   waypoints: waypts,          //the waypts array is created by the createWaypoints function and contains waypoint object to be added to the route
   optimizeWaypoints: true,
   travelMode: google.maps.TravelMode.DRIVING
  }
  function displayDirections(response, status) { //this function is supposed to change the alert banner and place markers on locations passed in from the response object
   if (status === google.maps.DirectionsStatus.OK) {
    document.getElementById("final_span").innerHTML = "Click on the Map or press Record to add a new place,<br>otherwise click 'Start Over'";
    changeAlertBanner("#4CAF50", "<strong>Success!</strong> Click on a Marker to get your trip information!");
    directionsDisplay.setDirections(response); //supposed to take the response and make a polyline out of it
    directionsDisplay.setMap(map);             //puts said polyline on the map
    document.getElementById("directions").onclick = function(){setDirectionsLink(response)};
    removeMarkers();
    for (var i=1; i<response.routes[0].legs.length; i++){
     var marker = new google.maps.Marker({
      map:map,
      position:response.routes[0].legs[i].start_location
     });
     google.maps.event.addListener(marker, "click", openNav);
     markers.push(marker);
    }
   } 
   else {//if these is an error placing finding a route then placeMarkersOnError is called to place 
      placeMarkersOnError(placeArray); //markers on the places that can be found with the geocoding service
      document.getElementById("final_span").innerHTML = "Click on the Map or press Record to add a new place,<br>otherwise click 'Start Over'";
      changeAlertBanner("#f44336", "No directions could be found connecting all locations, Click on Markers for more place information")
     }  
    }   

  directionsService.route(places, displayDirections); //supposed to make a route out of the perviously declared places object
                                               //and set it on the map with mrkers added using a callback to displayDirections
}


//helper function for the calculeAndDisplayRoute
//intended to turn all the locations passed in into an array of waypoints objects
//HR
function createWaypoints(places) {
 waypts = [];
 for (var i=0; i<places.length; i++) {
  waypts.push({location: places[i], stopover: true});
 }
 if (waypts.length > 8) {
  waypts = waypts.slice(0, 8);
 }
}

//this function is intended to be called when the map is clicked and adds a marker on that spot
function addMarker(event) {
 var marker = new google.maps.Marker({
  map:map,
  position:event.latLng
 });
 google.maps.event.addListener(marker, "click", openNav);
 markers.push(marker);
 placeArray.push(event.latLng);
 clickArray.push(event.latLng); //the clickArray stores any locations the user has added by clicking on the map,
                               //it is needed because placesArray gets cleared if the user adds another location by voice
 if(placeArray.length > 1 && !ignore_removeMarkers) { //this if statement makes sure there are at least 2 locations added before 
  createWaypoints(placeArray);                        //calculating a route with new markers and removing the old makers
  calculateAndDisplayRoute(directionsService, directionsDisplay); 
  
 } 
}

//supposed to look at all the markers that have been stored in the markers array and gets rid of them when called
//HR
function removeMarkers() {
 for (var i = 0; i < markers.length; i++) {
    markers[i].setMap(null);
  }
  markers = [];
}

//if there is an error finding a route when calculateAndDisplayRoute runs then placeMarkersOnError is called to put 
function placeMarkersOnError(places) { //markers on the places that can be found with the geocoding service
 geocoder = new google.maps.Geocoder()
 bounds = new google.maps.LatLngBounds();
 //HR
 for (var i=0; i<places.length; i++) {
 geocoder.geocode({'address': places[i]}, function(results, status) {
    if (status === google.maps.GeocoderStatus.OK) {
      map.setCenter(results[0].geometry.location);
      var marker = new google.maps.Marker({
        map: map,
        position: results[0].geometry.location
      });
      google.maps.event.addListener(marker, 'click', openNav);
        markers.push(marker);
       placeArray.push(results[0].geometry.location);
       
       ignore_removeMarkers = true;
    } 
    else {
      changeAlertBanner("#f44336","'Geocode could not find that location...");
    }
    for (var i = 0; i < markers.length; i++) {
     bounds.extend(markers[i].getPosition());
    }
    map.fitBounds(bounds);
  });
 }
}


//Navigation overlay functions
function openNav(e) {
    loadImages(e.latLng.lat(), e.latLng.lng()); //loads images in the display window when the navigation is opened
    document.getElementById("navLayer").style.height = "100%";
    document.getElementById("displayWindow").innerHTML = "Loading...";
    document.getElementById("photos").onclick = function(){setPhotosLink(e)};
    document.getElementById("food").onclick = function(){setFoodLink(e)};
    document.getElementById("stay").onclick = function(){setStayLink(e)};
}

//these functions are set to be called when the navigation links are clicked, get filled with location information stored in the clicked marker
function setPhotosLink(e) {
 loadImages(e.latLng.lat(), e.latLng.lng())
}

function setDirectionsLink(response) {
 var tripText = formatTripDetails(response);
 document.getElementById("displayWindow").innerHTML = tripText;
}

function setFoodLink(e) {
loadPlaces(e.latLng.lat(), e.latLng.lng(), 'restaurant');
}

function setStayLink(e) {
loadPlaces(e.latLng.lat(), e.latLng.lng(), 'lodging');
}


function closeNav() {
    document.getElementById("navLayer").style.height = "0%";
    return false;
}


//Photo search helper functions
function loadImages(lat, lon) {
 var xhr = new XMLHttpRequest();
 xhr.onreadystatechange = function() {
  if (xhr.readyState == 4 && xhr.status == 200) {
   var picHTML = displayPicsFromData(25, JSON.parse(xhr.responseText));
   document.getElementById("displayWindow").innerHTML = picHTML;
  }
  else if (xhr.status == 404) {
   document.getElementById("displayWindow").innerHTML = "There was an error...";
  };
 };
 xhr.open("GET", "https://api.flickr.com/services/rest/?method=flickr.photos.search&" + 
  "api_key=9edf145e07ba220c42ade82e1759810d&lat=" + lat + "&lon=" + lon + 
  "&accuracy=1&sort=interestingness-desc&extras=url_l&format=json&nojsoncallback=1", true);
 xhr.send();
}


function displayPicsFromData(numImages, data) { //formats a specified number of images from the passed in object into an html string 
 html = "<ul>";
  for (var i=0; i<numImages; i++){
   html += "<li><a href='https://www.flickr.com/photos/" + data.photos.photo[i].owner + 
   "/" + data.photos.photo[i].id + "' target='_blank'><img id='pic' src='" + data.photos.photo[i].url_l + "'></a></li>"
  };
  html += "</ul>";
  return html;
}


//Trip Detail helper functions 
function formatTripDetails(data){ //called when directions are returned or 'trip detail' is clicked in navigation panel
 var totalDistance = 0;           //but the results can only be seen when navigation panel is open
 var tripText = "";
 for(var i=0; i<data.routes[0].legs.length; i++){
  if (data.routes[0].legs[i].distance.value != 0){  //only displays details for waypoints not on top of eachother
   totalDistance += data.routes[0].legs[i].distance.value;
   tripText += "<strong>FROM</strong> " + data.routes[0].legs[i].start_address + " <strong>TO</strong> " +
                data.routes[0].legs[i].end_address + ":<br><u>" + data.routes[0].legs[i].distance.value/1000 + 
                " Kilometers</u><br><br>";
  }
 }
 tripText = "Total Round Trip Distance:<br><strong>" + totalDistance/1000 + " Kilometers</strong><br><br>" + tripText;
 return tripText;
}


//Food and Stay search helper functions
function loadPlaces(lat, lng, placeType) { //searches for a certain type of place around the passed in coordinates
 var service = new google.maps.places.PlacesService(map);
 service.nearbySearch({
  location: {lat: lat, lng: lng},
  radius: 5000,
  type: [placeType]
 }, formatPlaceData);
}


function formatPlaceData(results) { //a callback that takes the place search object and formats its infotmation  
 var numPlaces;                     //so it looks nice to display. used to search for lodging and resturants
 html = "<ul>";
 if (results.length > 10){ //if there are more than 10 results it only shows 10 max
   numPlaces = 10;
 }
 else{
   numplaces = results.length;
 }//HR
 for (var i=0; i<numPlaces; i++){
  html += "<li><u><a href='" + makeAddressSearchable(results[i].vicinity, results[i].name) + "' target='_blank'>" + 
  results[i].name + "</a></u>" + results[i].rating + " stars<br><br></li><br>"
 };
 html += "</ul>";
 document.getElementById("displayWindow").innerHTML = html ;
}

function makeAddressSearchable(adrString, nameString){//uses the vicinity and name properties of the object to make a 
 address = "https://www.google.com/search?q="         //link that searches for the place when the name is clicked
 var adrComponents = adrString.split(", ");
 var nameComponents = nameString.split(" ");
 for (var i=0; i<adrComponents.length; i++){
   address += adrComponents[i] + "+";
 }
 for (var i=0; i<nameComponents.length; i++){
   address += nameComponents[i] + "+";
 }
 return address;
}


//Functions that modify the transcript after speack recognition
//HR
function modifyPlaceNames(fixes, array){
 var a = array;
 for (var i=0; i<a.length; i++){       //this function is interesting, if the user says a place like 'new york' or 'cannon beach', 
   for (var j=0; j<fixes.length; j++){ //instead of searching for 'new' and 'york' or 'cannon' and 'beach', it looks at the 
   if (a[i] == fixes[j]){              //prefixes and postfixes passes in and decides if it should concatenate and how
    if (fixes[0] == 'pre'){
    a.splice(i, 2, fixes[j] + " " + a[i+1]);// this is supposed to add prefixes
    }
    else {a.splice(i-1, 2, a[i-1] + " " + fixes[j]);}//this is supposed to add postfixes
   }
  }
 }
 return a;
}
//HR
function cleanResults(currentValue) { //this function is used as the callback that tests if a value should be removed by the filter method
 var pass = true;                     //it gets rid of words i decided to be 'filler words' and tells filter to remove them from the array
 var bannedWords = fillerWords
  for (var j=0; j<bannedWords.length; j++){
   if (currentValue == bannedWords[j]){
    pass = false;
   }
  }
  return pass;
}

//function that change the color and message of the alert banner at on the page
function changeAlertBanner(color, message) {
 document.getElementById("alert").innerHTML = message;
 document.getElementById("alert").style.backgroundColor = color;
}

//called when the 'start over' button is pressed and sets all global variables back to starting state
function startOver() {
 removeMarkers();
 ignore_removeMarkers = false;
 final_transcript = '';
 placeArray = [];
 clickArray = [];
 waypts = [];
 directionsDisplay.setMap(null);
 document.getElementById("final_span").innerHTML = "Click Record and name<br>up to 8 cities near each other";
 document.getElementById("interim_span").innerHTML = "";
 changeAlertBanner("#2196F3", "Or click on the Map to add places")
}


//checks if your browser supports the speech API
if (!('webkitSpeechRecognition' in window)) {
  start_img.src = 'noMic.gif';
  changeAlertBanner(" #f44336", "Looks like there was a problem...");
  alert("Sorry, your Browser does not support the Speech API. Try using Google Chrome");
} 
else {
  var recognition = new webkitSpeechRecognition(); //creates a recognition object and sets its properties
  recognition.lang = 'en-US'; //sets the accent to an american accent
  recognition.continuous = true;
  recognition.interimResults = true;

  //callback hooks that deal with the various states the recognition object can be in
  recognition.onstart = function() { //when the recognition object starts listening
    recognizing = true;
    start_img.src = 'record.gif';
    changeAlertBanner("#4CAF50", "Press the Record button when finished...");
  };
  
  recognition.onerror = function(event) {//when the recognition object throws an error. some common types are handled here
    if (event.error == 'not-allowed') {
      start_img.src = 'noMic.gif'
      changeAlertBanner(" #f44336", "Looks like there was a problem...");
      alert("Permission to use microphone was denied.");
      ignore_onend = true;
    }
    if (event.error == 'audio-capture') {
      start_img.src = 'noMic.gif'
      changeAlertBanner(" #f44336", "Looks like there was a problem...");
      alert(" No microphone was found. Ensure that a microphone is installed and that microphone settings are configured correctly."); 
      ignore_onend = true;
    }
  };
  
  recognition.onend = function() { //when the recognition object stops listening
    recognizing = false;
    if (placeArray.length == 0){  //if it didnt hear anthing the placeArray will be empty and asks to try again
    document.getElementById("final_span").innerHTML = "Sorry, I didnt quite catch that... Try Again?";
    start_img.src = 'noMic.gif'
    }
    else {
    placeArray = placeArray.concat(clickArray) //clickArray stores location data of markers added by clicking the map
    createWaypoints(placeArray);             //this makes sure those are included in direction calculations
    calculateAndDisplayRoute(directionsService, directionsDisplay);
    }
    start_img.src = 'click.gif';
  };
  
  recognition.onresult = function(event) { //while the recognition object is listening it returns guesses in its resultIndex
    var interim_transcript = '';        
    for (var i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        final_transcript += event.results[i][0].transcript; //if its sure thats what you said, shove it in the final_transcript variable
      } 
      else {
        interim_transcript += event.results[i][0].transcript; //if its thinks thats what you said, shove it in the interim_transcript variable
      }
    }
    final_transcript = capitalize(final_transcript);
    final_span.innerHTML = linebreak(final_transcript);
    interim_span.innerHTML = linebreak(interim_transcript);
    final_transcript = final_transcript.toLowerCase();
    
    removeMarkers();
    placeArray = final_transcript.split(' ').filter(cleanResults)
    placeArray = modifyPlaceNames(prefixes, placeArray);
    placeArray = modifyPlaceNames(postfixes, placeArray);
  };
var two_line = /\n\n/g;
var one_line = /\n/g;
function linebreak(s) {
  return s.replace(two_line, '<p></p>').replace(one_line, '<br>');
}
var first_char = /\S/;
function capitalize(s) {
  return s.replace(first_char, function(m) { return m.toUpperCase(); });
}

//this function is called when the record button is clicked and makes the recognition object either start or stop listening
function startButton() {
  if (recognizing) {
    recognition.stop();
    return;
  }
  final_transcript += ' '; //makes it so you can start recording again and it will add the next results to the last results
  placeArray = [];          // as long as you didnt invoke the startOver function
  waypts = [];
  recognition.start();
  ignore_onend = false;
  final_span.innerHTML = '';
  interim_span.innerHTML = '';
 }
}
