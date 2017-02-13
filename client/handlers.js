    //get it to console.log somethin on click by including this file
    //add a method on skyctrl that console logs something
    //add gruntfile for linting and commiting
	//hook the cube to the skycontroller to call the method
	//make the skyctrolr make a request
	//rerender the sky 
//make some tests ?
//allow login (use angular routes)

$(document).ready(function() {
//when box is clicked, change to red
  $('#box').on('click', function (evt) {
    var tags = document.getElementsByClassName('tag')
    tags = Array.prototype.slice.call(tags);
    tags.forEach(function(tag) {
    console.log(tag);
    tag.setAttribute('visible', false);
  })
    //see if you can permenantly change the color so it can toggle
  });

});

