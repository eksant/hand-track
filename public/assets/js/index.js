// import { load } from './handTrack';

// import * as handTrack from './handTrack';

let video = document.getElementById('myvideo');
let handimg = document.getElementById('handimage');
let canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

let trackButton = document.getElementById('trackbutton');
let updateNote = document.getElementById('updatenote');

let imgindex = 1;
let isVideo = false;
let model = null;

const modelParams = {
  flipHorizontal: true,
  maxNumBoxes: 20,
  iouThreshold: 0.5,
  scoreThreshold: 0.6,
};

handTrack.load(modelParams).then(function(lmodel) {
  model = lmodel;
  updateNote.innerText = 'Loaded Model!';
  trackButton.disabled = false;
});

trackButton.addEventListener('click', function() {
  toggleVideo();
});

function toggleVideo() {
  if (!isVideo) {
    updateNote.innerText = 'Starting video';
    startVideo();
  } else {
    updateNote.innerText = 'Video stopped';
    handTrack.stopVideo(video);
    isVideo = false;
    updateNote.innerText = 'Video stopped';
  }
}

function startVideo() {
  handTrack.startVideo(video).then(function(status) {
    console.log('video started', status);
    if (status) {
      updateNote.innerText = 'Video started. Now tracking';
      isVideo = true;
      runDetection();
    } else {
      updateNote.innerText = 'Please enable video';
    }
  });
}

function runDetection() {
  model.detect(video).then(predictions => {
    // console.log('Predictions: ', predictions);
    model.renderPredictions(predictions, canvas, context, video);
    if (isVideo) {
      requestAnimationFrame(runDetection);
    }
  });
}
