const TOTAL = 500;
let generation = 0;

let birds = [];
let pipes = [];
let playing = false;
let counter = 0;
let episodes = 10;
let slider = 1;
let bestScore = 0;
tf.setBackend("cpu");

function preload() {
  birdImg = loadImage("./assets/bird.png");
  pipeImg = loadImage("./assets/pipes.png");
  pipeRevImg = loadImage("./assets/pipes_reverse.png");
  backgroundImg = loadImage("./assets/background.png");
}

function setup() {
  createCanvas(400, 600);
  frameRate(40);
  slider = createSlider(1, 100, 1);
  angleMode(DEGREES);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  textSize(50);

  for (let index = 0; index < TOTAL; index++) {
    birds.push(new Bird());
  }
}

function draw() {
  for (let index = 0; index < slider.value(); index++) {
    if (birds.some((bird) => bird.locked)) {
      text("Training", width / 2, height / 5);
      return;
    }

    if (counter % 40 == 0) {
      pipes.push(new Pipe());
    }

    const livedBirds = birds.filter((bird) => !bird.die);

    livedBirds.forEach((bird) => {
      bird.frames++;
    });

    for (let i = pipes.length - 1; i >= 0; i--) {
      pipes[i].update();

      if (pipes[i].offScreen()) {
        livedBirds.forEach((bird) => {
          if (pipes[i].pass(bird)) {
            bird.score++;
          }
        });
        pipes.splice(i, 1);
      }

      livedBirds.forEach((bird) => {
        if (pipes[i].hit(bird)) {
          bird.done();
        }
      });
    }

    // draw bird
    livedBirds.forEach((bird) => {
      bird.think(pipes);

      bird.update();
    });

    if (livedBirds.length === 0) {
      // New Generation
      pipes = [];

      const sortedBirds = birds.sort((a, b) => a.frames - b.frames);

      const bestBird = birds[birds.length - 1];

      nextGeneration(birds);

      // bestBird?.train().then((aa) => {
      //   sortedBirds.forEach((bird, list) => bird.reset(aa));
      // });
    }

    counter++;
  }

  // Draw Here
  background(backgroundImg);

  pipes.forEach((pipe) => {
    pipe.show();
  });

  birds
    .filter((bird) => !bird.die)
    .slice(0, 100)
    .forEach((bird) => {
      bird.show();
    });

  const maxScore = Math.max(...birds.map((bird) => bird.score));

  if (maxScore > bestScore) {
    bestScore = maxScore;
  }
  text(`${maxScore}:${bestScore}`, width / 2, height / 5);
}
