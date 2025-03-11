// Header Fractal Plants Animation
// Script che genera frattali simili a piante che crescono e scompaiono nell'header

let plants = [];
const MAX_PLANTS = 12;
const GRASS_COLOR = '#1a5c1a'; // Verde erba scuro
const PLANT_COLOR = '#d0ffd0'; // Verde chiarissimo per i frattali

function setup() {
  // Crea un canvas che si adatta al contenitore dell'header
  let headerElement = document.getElementById('fractal-canvas-header');
  let canvas = createCanvas(headerElement.offsetWidth, headerElement.offsetHeight);
  canvas.parent('fractal-canvas-header');
  
  // Imposta lo sfondo verde erba
  background(GRASS_COLOR);
  
  // Inizializza alcune piante
  for (let i = 0; i < 5; i++) {
    addNewPlant();
  }
}

function draw() {
  // Ridisegna lo sfondo con una leggera trasparenza per creare un effetto di dissolvenza
  background(GRASS_COLOR + '10'); // Aggiunge una leggera trasparenza
  
  // Aggiorna e disegna tutte le piante
  for (let i = plants.length - 1; i >= 0; i--) {
    let plant = plants[i];
    
    // Aggiorna lo stato della pianta
    plant.age += 0.015; // Velocità di crescita leggermente ridotta
    
    // Se la pianta è troppo vecchia, rimuovila
    if (plant.age > plant.lifespan) {
      plants.splice(i, 1);
      // Aggiungi una nuova pianta se ne abbiamo rimosse alcune
      if (random() > 0.6 && plants.length < MAX_PLANTS) {
        addNewPlant();
      }
      continue;
    }
    
    // Disegna la pianta
    drawPlant(plant);
  }
  
  // Aggiungi nuove piante casualmente
  if (random() > 0.97 && plants.length < MAX_PLANTS) {
    addNewPlant();
  }
}

function addNewPlant() {
  plants.push({
    x: random(width),
    y: height + random(10, 30), // Inizia leggermente sotto il bordo inferiore
    size: random(30, 80),
    angle: -HALF_PI + random(-0.2, 0.2), // Principalmente verso l'alto con leggera variazione
    branches: floor(random(3, 6)),
    depth: floor(random(3, 5)),
    age: 0,
    lifespan: random(5, 10),
    growthRate: random(0.4, 1.2),
    type: floor(random(3)) // Tipo di pianta (0, 1, 2)
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
  strokeWeight(1.5);
  noFill();
  
  // Disegna la pianta frattale
  push();
  translate(plant.x, plant.y);
  rotate(plant.angle);
  
  // Diversi tipi di piante
  switch(plant.type) {
    case 0:
      drawFern(currentSize, plant.branches, plant.depth);
      break;
    case 1:
      drawBranch(currentSize, plant.branches, plant.depth);
      break;
    case 2:
      drawFlower(currentSize, plant.branches, plant.depth);
      break;
  }
  
  pop();
}

// Disegna un ramo standard
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

// Disegna una felce
function drawFern(len, branches, depth) {
  if (depth <= 0) return;
  
  // Disegna il ramo principale
  line(0, 0, 0, -len);
  
  // Disegna le foglie laterali
  let leafCount = floor(len / 10);
  for (let i = 1; i <= leafCount; i++) {
    let y = -i * (len / leafCount);
    
    // Foglia sinistra
    push();
    translate(0, y);
    rotate(-QUARTER_PI - random(0.2));
    let leafSize = len * 0.3 * (1 - i/leafCount * 0.5);
    line(0, 0, leafSize, 0);
    pop();
    
    // Foglia destra
    push();
    translate(0, y);
    rotate(QUARTER_PI + random(0.2));
    leafSize = len * 0.3 * (1 - i/leafCount * 0.5);
    line(0, 0, leafSize, 0);
    pop();
  }
}

// Disegna un fiore
function drawFlower(len, petals, depth) {
  if (depth <= 0) return;
  
  // Disegna lo stelo
  line(0, 0, 0, -len);
  translate(0, -len);
  
  // Disegna i petali
  let angleStep = TWO_PI / petals;
  for (let i = 0; i < petals; i++) {
    push();
    rotate(i * angleStep);
    beginShape();
    vertex(0, 0);
    bezierVertex(
      len * 0.3, -len * 0.2,
      len * 0.5, -len * 0.3,
      0, -len * 0.5
    );
    endShape();
    pop();
  }
}

function windowResized() {
  let headerElement = document.getElementById('fractal-canvas-header');
  resizeCanvas(headerElement.offsetWidth, headerElement.offsetHeight);
  background(GRASS_COLOR);
}
