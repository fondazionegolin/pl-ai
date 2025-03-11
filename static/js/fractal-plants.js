// Fractal Plants Animation
// Script che genera frattali simili a piante che crescono e scompaiono

let plants = [];
const MAX_PLANTS = 15;
const GRASS_COLOR = '#1a5c1a'; // Verde erba scuro
const PLANT_COLOR = '#c0ffc0'; // Verde chiarissimo per i frattali

function setup() {
  // Crea un canvas che copre l'intera finestra
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.position(0, 0);
  canvas.style('z-index', '-2');
  
  // Imposta lo sfondo verde erba
  background(GRASS_COLOR);
  
  // Inizializza alcune piante
  for (let i = 0; i < 5; i++) {
    addNewPlant();
  }
}

function draw() {
  // Ridisegna lo sfondo con una leggera trasparenza per creare un effetto di dissolvenza
  background(GRASS_COLOR + '15'); // Aggiunge una leggera trasparenza
  
  // Aggiorna e disegna tutte le piante
  for (let i = plants.length - 1; i >= 0; i--) {
    let plant = plants[i];
    
    // Aggiorna lo stato della pianta
    plant.age += 0.02;
    
    // Se la pianta è troppo vecchia, rimuovila
    if (plant.age > plant.lifespan) {
      plants.splice(i, 1);
      // Aggiungi una nuova pianta se ne abbiamo rimosse alcune
      if (random() > 0.7 && plants.length < MAX_PLANTS) {
        addNewPlant();
      }
      continue;
    }
    
    // Disegna la pianta
    drawPlant(plant);
  }
  
  // Aggiungi nuove piante casualmente
  if (random() > 0.98 && plants.length < MAX_PLANTS) {
    addNewPlant();
  }
}

function addNewPlant() {
  plants.push({
    x: random(width),
    y: random(height),
    size: random(30, 100),
    angle: random(TWO_PI),
    branches: floor(random(3, 6)),
    depth: floor(random(3, 5)),
    age: 0,
    lifespan: random(5, 10),
    growthRate: random(0.5, 1.5)
  });
}

function drawPlant(plant) {
  // Calcola la dimensione attuale in base all'età
  let growthFactor;
  
  // Crescita e dissolvenza
  if (plant.age < plant.lifespan * 0.7) {
    // Fase di crescita: da 0 a 1
    growthFactor = min(1, plant.age * plant.growthRate);
  } else {
    // Fase di dissolvenza: da 1 a 0
    growthFactor = map(plant.age, plant.lifespan * 0.7, plant.lifespan, 1, 0);
  }
  
  let currentSize = plant.size * growthFactor;
  
  // Imposta il colore con opacità basata sull'età
  let opacity = 255 * growthFactor;
  stroke(red(color(PLANT_COLOR)), green(color(PLANT_COLOR)), blue(color(PLANT_COLOR)), opacity);
  strokeWeight(1);
  noFill();
  
  // Disegna la pianta frattale
  push();
  translate(plant.x, plant.y);
  rotate(plant.angle);
  drawBranch(currentSize, plant.branches, plant.depth);
  pop();
}

function drawBranch(len, branches, depth) {
  if (depth <= 0) return;
  
  // Disegna il ramo principale
  line(0, 0, 0, -len);
  translate(0, -len);
  
  // Disegna i rami secondari
  let angleStep = TWO_PI / branches;
  for (let i = 0; i < branches; i++) {
    push();
    rotate(i * angleStep);
    drawBranch(len * 0.7, branches - 1 + floor(random(-1, 1)), depth - 1);
    pop();
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  background(GRASS_COLOR);
}
