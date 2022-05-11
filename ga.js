function pickOne(birds) {
  let index = 0;
  let r = Math.random();

  while (r > 0) {
    r -= birds[index].fitness;
    index += 1;
  }

  index--;

  return birds[index];
}

function crossover(parentA, parentB) {
  return tf.tidy(() => {
    const weightsA = parentA.brain.getWeights();
    const weightsB = parentB.brain.getWeights();

    return [
      weightsA.map((wA, index) => {
        const wB = weightsB[index];
        const dataA = wA.dataSync();
        const dataB = wB.dataSync();

        const mid = Math.floor(random(0, dataA.length));
        const newWeight = [
          ...dataA.slice(0, mid),
          ...dataB.slice(mid, dataA.length),
        ];

        return tf.tensor(newWeight, wA.shape);
      }),

      weightsA.map((wA, index) => {
        const wB = weightsB[index];
        const dataA = wA.dataSync();
        const dataB = wB.dataSync();

        const mid = Math.floor(random(0, dataA.length));

        const newWeight = [
          ...dataB.slice(0, mid),
          ...dataA.slice(mid, dataA.length),
        ];

        return tf.tensor(newWeight, wA.shape);
      }),
    ];
  });
}

function mutate(weights) {
  return tf.tidy(() => {
    return weights.map((weight) => {
      const data = weight.dataSync();
      const newData = data.map((value) =>
        Math.random() > 0.1 ? value : value + randomGaussian() * 0.5
      );

      return tf.tensor(newData, weight.shape);
    });
  });
}

function nextGeneration(birds) {
  const sum = birds.reduce((score, bird) => score + bird.score, 0);
  birds.forEach((bird) => (bird.fitness = bird.score / sum));
  const sortedBirds = birds.sort((a, b) => b.frames - a.frames);

  const [parentA, parentB] = sortedBirds;
  tf.tidy(() => {
    const [weightsA, weightsB] = crossover(parentA, parentB);

    birds.forEach((bird, index) => {
      if (index === 0) {
        //   console.log(bird.brain.getWeights(), weightsA);
        bird.brain.setWeights(weightsA);
      } else if (index === 1) {
        bird.brain.setWeights(weightsB);
      } else if (Math.random() > 0.5) {
        bird.brain.setWeights(mutate(weightsA));
      } else {
        bird.brain.setWeights(mutate(weightsB));
      }

      bird.die = false;
      bird.score = 0;
      bird.frames = 0;
      bird.x = width / 3;
      bird.y = height / 3;
    });
  });
}
