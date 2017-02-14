var app = angular.module('teleporter', []); //important to put in a var

app.controller('SkyCtrl', function($scope, $http, $sce) {
  
  $scope.parseUrl = function(url) {
    return JSON.parse(JSON.stringify(url));
  };

  $scope.equirectangular = 'https://c6.staticflickr.com/9/8273/30215908325_2cacc343fa_b.jpg';

  $scope.tag = 'not happening';

  $scope.tagsOn = false;

  $scope.trustSrc = function(src) {
    return $sce.trustAsResourceUrl(src);
  };

  $scope.fetchNewImage = function() {
    console.log('FETCHING...');
    console.log($scope.tag);
    return $http({
      url: 'https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key=9edf145e07ba220c42ade82e1759810d&tags=' + $scope.tag + '&group_id=44671723%40N00&extras=url_l&format=json&nojsoncallback=1',
      method: 'GET'
    }).then(function(res) {
      var rand = Math.floor(Math.random() * res.data.photos.photo.length);
      $scope.equirectangular = $scope.parseUrl(res.data.photos.photo[rand].url_l);
      console.log($scope.equirectangular);
      $scope.toggleTags();
    });
  };

  $scope.toggleTags = function() {
    var tags = document.getElementsByClassName('tag')
    tags = Array.prototype.slice.call(tags);
    tags.forEach(function(tag) {
      if ($scope.tagsOn){
        tag.setAttribute('visible', false);
      } else{
        tag.setAttribute('visible', true);
      }
    });
   $scope.tagsOn = !$scope.tagsOn 
  }

  // $scope.$watch('equirectangular', function() {
  //   $scope.skysrc = 'pic';
  //   $scope.skysrc = '#pic';
  // });

});

