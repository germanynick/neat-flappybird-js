class Bird {
  constructor() {
    this.x = width / 3;
    this.y = height / 3;
    this.r = 40;
    this.gravity = 2;
    this.velocity = 0;
    this.lift = -20;
    this.friction = 0.2;
    this.score = 0;
    this.frames = 0;
    this.up = false;
    this.die = false;
    this.brain = this.createBrain();
    this.target_brain = this.createBrain();

    this.states = [];
    this.actions = [];
    this.scores = [];

    this.trained = false;
    this.locked = false;
  }

  show() {
    fill(255);
    push();
    imageMode(CENTER);
    translate(this.x, this.y);
    if (this.up || this.velocity < 0) {
      rotate(-35);
    } else {
      rotate(35);
    }
    // ellipse(this.x, this.y, this.r);
    image(birdImg, 0, 0, this.r, this.r);
    pop();
  }

  update() {
    this.velocity += this.gravity;
    this.velocity = constrain(this.velocity, -25, 25);
    this.y += this.velocity;
    if (this.y > height) {
      this.velocity = 0;
      this.y = height;
    } else if (this.y < 0) {
      this.velocity = 0;
      this.y = 0;
    }
    this.up = false;
  }

  flap() {
    this.velocity += this.lift;
    this.velocity *= 1 - this.friction;
    this.up = true;
  }

  think(pipes) {
    const notPassed = pipes.filter((pipe) => !pipe.pass(this));

    const closest = notPassed[0];
    const next = notPassed[1];

    if (closest) {
      tf.tidy(() => {
        const state = [
          this.y / height,
          closest.top / height,
          closest.x / width,
          (next?.top || 0) / height,
          (next?.x || 0) / width,
        ];

        const xs = tf.tensor([state]);

        const ys = this.brain.predict(xs);

        const action = ys.argMax(1).dataSync().at(0);
        // ys.print();

        this.states.push(state);
        this.actions.push(action);
        this.scores.push(this.score);

        if (action === 1) {
          this.flap();
        }
      });
    }
  }

  createBrain() {
    const brain = tf.sequential();
    brain.add(
      tf.layers.dense({ units: 48, inputShape: [5], activation: "tanh" })
    );
    brain.add(tf.layers.dropout({ rate: 0.2 }));
    brain.add(tf.layers.dense({ units: 24, activation: "tanh" }));
    brain.add(tf.layers.dropout({ rate: 0.2 }));
    brain.add(tf.layers.dense({ units: 2, activation: "tanh" }));

    brain.compile({
      optimizer: "sgd",
      loss: "meanSquaredError",
      metrics: ["acc"],
    });

    // brain.summary();

    return brain;
  }

  done() {
    this.die = true;
  }

  reset(bird) {
    this.die = false;
    this.score = 0;
    this.x = width / 3;
    this.y = height / 3;
    this.frames = 0;

    this.mutate(bird);

    this.states = [];
    this.actions = [];
    this.trained = false;
  }

  async train() {
    return Promise.resolve(this);

    return new Promise((resolve) => {
      this.locked = true;

      const { xs, ys } = tf.tidy(() => {
        const xs = tf.tensor(this.states.slice(0, this.states.length - 1));
        const xss = tf.tensor(this.states.slice(1, this.states.length));

        const ys = this.brain.predict(xs);
        const yss = this.brain.predict(xss);
        const ysss = this.target_brain.predict(xss);

        const q_values = ys.bufferSync();
        const next_actions = yss.argMax(1).dataSync();
        const next_q_values = ysss.bufferSync();

        for (let index = 0; index < this.states.length - 1; index++) {
          const action = this.actions[index];
          let reward = index / 1000;

          if (index > this.states.length - 5) {
            reward = -reward;
          }

          // console.log(reward);

          const next_action = next_actions.at(index);
          const next_q_value = next_q_values.get(index, next_action);

          const expect_q_value = reward + 0.95 * next_q_value;

          q_values.set(expect_q_value, index, action);
        }

        return { xs, ys: q_values.toTensor() };
      });

      ys.print();

      this.brain.fit(xs, ys, {
        epochs: 10,
        callbacks: {
          onTrainEnd: () => {
            this.locked = false;
            this.trained = true;
            resolve(this);
          },
        },
      });
    });
  }

  async mutate(bird) {
    tf.tidy(() => {
      const current = this.brain.getWeights();
      const target = bird.brain.getWeights();

      const weights = target.map((weight, index) => {
        const temp = current[index];
        const dataA = weight.dataSync();
        const dataB = temp.dataSync();

        const mid = Math.floor(random(0, dataA.length));

        // const newWeight = dataA.map((value, index) =>
        //   Math.random() < 0.5 ? value : dataB[index]
        // );

        const newWeight = [
          ...dataA.slice(0, mid),
          ...dataB.slice(mid, dataA.length),
        ];

        return tf.tensor(newWeight, weight.shape);
      });

      this.brain.setWeights(weights);
      this.target_brain.setWeights(weights);
    });
  }
}
