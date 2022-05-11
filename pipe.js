class Pipe {
  constructor() {
    this.x = width;
    this.w = 90;
    this.gap = 150;
    this.min_height = 100;
    this.max_height = height - this.min_height - this.gap;
    this.top = floor(random(this.min_height, this.max_height));
    this.speed = 10;
  }

  show() {
    fill(255, 0, 0);
    /* top pipe */
    // rect(this.x, 0, this.w, this.top);
    image(pipeRevImg, this.x, 0, this.w, this.top);
    /* bottom pipe */
    fill(0, 255, 0);
    let height_b = height - this.gap - this.top;
    let y_b = height - height_b;
    // rect(this.x, y_b, this.w, height_b);
    image(pipeImg, this.x, y_b, this.w, height_b);
  }

  offScreen() {
    if (this.x + this.w + this.speed < 0) {
      return true;
    }
    return false;
  }

  hit(bird) {
    if (bird.x > this.x && bird.x < this.x + this.w) {
      if (bird.y < this.top || bird.y > this.top + this.gap) {
        return true;
      }
    }
    return false;
  }

  pass(bird) {
    if (bird.x > this.x + this.w) {
      return true;
    }
    return false;
  }

  update() {
    this.x -= this.speed;
  }
}
