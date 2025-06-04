// JavaScript per machinelearning2.html

// Variabili globali
let rawData = [];
let headers = [];

// Event listener per il caricamento del file CSV
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('csv-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                console.log('File selezionato:', file.name);
                handleCsvFileUpload(file);
            }
        });
    } else {
        console.error('Elemento input file (#csv-file-input) non trovato');
    }

    // Aggiungi un event listener alla colonna target per debug
    const targetColumnSelect = document.getElementById('target-column-select');
    if (targetColumnSelect) {
        targetColumnSelect.addEventListener('change', function() {
            console.log('Evento change sulla colonna target rilevato.');
            suggestProblemType(this.value); // Chiamata esistente, solo log aggiuntivo qui
        });
    } else {
         console.error('Elemento select colonna target (#target-column-select) non trovato');
    }

    // Aggiungi event listener alla tendina del tipo di problema
    const problemTypeSelect = document.getElementById('problem-type-select');
    if (problemTypeSelect) {
        problemTypeSelect.addEventListener('change', function() {
            console.log('Evento change sul tipo di problema rilevato. Nuovo tipo:', this.value);
            populateAlgorithmSelect(this.value);
            displayAlgorithmParams(this.value, document.getElementById('algorithm-select')?.value); // Aggiorna parametri per il nuovo tipo di problema
            checkTrainButtonState();
        });
         console.log('Event listener aggiunto alla tendina Tipo di Problema.');
    } else {
         console.error('Elemento select tipo di problema (#problem-type-select) non trovato');
    }

    // Aggiungi event listener al pulsante di training
    const trainModelButton = document.getElementById('train-model-button');
    if (trainModelButton) {
        trainModelButton.addEventListener('click', handleModelTraining);
        console.log('Event listener aggiunto al pulsante Addestra Modello.');
    } else {
        console.error('Pulsante Addestra Modello (#train-model-button) non trovato');
    }
});

// Funzione per verificare se PapaParse è disponibile
function checkPapaParse() {
    console.log('Verifica PapaParse...');
    if (typeof Papa === 'undefined') {
        console.error('PapaParse non è stato caricato correttamente');
        alert('Errore: La libreria PapaParse non è stata caricata. Ricarica la pagina.');
        return false;
    }
    console.log('PapaParse è disponibile');
    return true;
}

// Funzione per visualizzare l'anteprima dei dati
function displayDataPreview(data, headers) {
    console.log('Chiamata displayDataPreview.');
    const previewContainer = document.getElementById('data-preview-table');
    if (!previewContainer) {
        console.error('Container per l\'anteprima dati non trovato');
        return;
    }

    // Pulisci il container di anteprima (tabella e visualizzazioni)
    previewContainer.innerHTML = '';

    // Rimuovi i grafici precedenti (sia 2D che 3D e matrice)
    const existingPlot2D = document.getElementById('plotly-2d-scatter');
    if (existingPlot2D) { Plotly.purge('plotly-2d-scatter'); existingPlot2D.parentNode.removeChild(existingPlot2D); }
    const existingPlot3D = document.getElementById('plotly-3d-scatter');
    if (existingPlot3D) { Plotly.purge('plotly-3d-scatter'); existingPlot3D.parentNode.removeChild(existingPlot3D); }
     const existingPlot2DContainer = document.getElementById('2d-scatter-plot-container'); // Rimuovi anche i container custom se esistono
    if (existingPlot2DContainer) { existingPlot2DContainer.parentNode.removeChild(existingPlot2DContainer); }
    const existingPlot3DContainer = document.getElementById('3d-scatter-plot-container');
    if (existingPlot3DContainer) { existingPlot3DContainer.parentNode.removeChild(existingPlot3DContainer); }


    const existingMatrix = document.getElementById('correlation-matrix');
    if (existingMatrix) { existingMatrix.parentNode.removeChild(existingMatrix); }

    // Crea e aggiungi la tabella di anteprima
    let table = document.createElement('table');
    table.className = 'min-w-full divide-y divide-gray-200';

    // Crea l'header
    let thead = document.createElement('thead');
    let headerRow = document.createElement('tr');
    headers.forEach(header => {
        let th = document.createElement('th');
        th.className = 'px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Crea il body
    let tbody = document.createElement('tbody');
    tbody.className = 'bg-white divide-y divide-gray-200';
    
    // Mostra solo le prime 5 righe
    data.slice(0, 5).forEach(row => {
        let tr = document.createElement('tr');
        headers.forEach(header => {
            let td = document.createElement('td');
            td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            td.textContent = row[header] || '';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    // Aggiungi la tabella al container di anteprima
    previewContainer.appendChild(table);


    // Trova le feature numeriche valide
    const numericFeatures = headers.filter(header => {
        const values = data.map(row => row[header]);
        // Controlla anche che non siano tutte stringhe che non possono essere convertite
        return values.every(val => !isNaN(parseFloat(val)) || (typeof val === 'string' && val.trim() === ''));
    });

     // Rimuovi eventuali feature numeriche che sono solo stringhe vuote
    const validNumericFeatures = numericFeatures.filter(feature => data.some(row => !isNaN(parseFloat(row[feature]))));

    console.log('displayDataPreview: Feature numeriche valide trovate:', validNumericFeatures);

    // Aggiungi i grafici in base al numero di feature numeriche
    if (validNumericFeatures.length >= 3) {
        // Aggiungi lo scatter plot 3D e la matrice di correlazione
        console.log('displayDataPreview: Trovate 3 o più feature numeriche, creo grafico 3D e matrice di correlazione.');
        create3dScatterPlot(data, headers); // create3dScatterPlot chiamerà createCorrelationMatrix
    } else if (validNumericFeatures.length === 2) {
        // Aggiungi il plot cartesiano 2D e la matrice di correlazione 2x2 (se rilevante)
        console.log('displayDataPreview: Trovate esattamente 2 feature numeriche, creo grafico 2D e matrice di correlazione.');
         create2dScatterPlot(data, validNumericFeatures);
         createCorrelationMatrix(data, validNumericFeatures); // Matrice di correlazione anche per 2 feature
    } else {
        console.log('displayDataPreview: Meno di 2 feature numeriche valide, nessun grafico 2D o 3D creato.');
        // Aggiungi un messaggio se non ci sono abbastanza feature numeriche per i grafici
         const visualizationSection = document.getElementById('data-visualization-section');
         if (visualizationSection) {
             const messageDiv = document.createElement('div');
             messageDiv.className = 'mt-8 p-4 bg-white rounded-lg shadow text-gray-700 text-center';
             messageDiv.textContent = 'Non ci sono abbastanza feature numeriche (almeno 2) per visualizzare grafici di dispersione.';
             visualizationSection.appendChild(messageDiv);
         }

         // Assicurati che la sezione visualizzazioni sia visibile anche senza grafici se ci sono dati validi
         const dataVisualizationSection = document.getElementById('data-visualization-section');
         if (dataVisualizationSection) dataVisualizationSection.classList.remove('hidden');
    }
}

// Funzione per creare un grafico di dispersione 2D
function create2dScatterPlot(data, numericFeatures) {
    console.log('Chiamata create2dScatterPlot con feature numeriche:', numericFeatures);
    if (numericFeatures.length < 2) {
        console.error('create2dScatterPlot richiede almeno 2 feature numeriche.');
        return;
    }

    // Trova o crea il container per il grafico 2D
    let plotContainer = document.getElementById('2d-scatter-plot-container');
    if (!plotContainer) {
        plotContainer = document.createElement('div');
        plotContainer.id = '2d-scatter-plot-container';
        plotContainer.className = 'mt-8 p-4 bg-white rounded-lg shadow';

        const title = document.createElement('h3');
        title.className = 'text-lg font-medium text-gray-900 mb-4';
        title.textContent = 'Visualizzazione 2D dei Dati';
        plotContainer.appendChild(title);

        // Crea il div per il grafico Plotly
        const plotDiv = document.createElement('div');
        plotDiv.id = 'plotly-2d-scatter';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '500px'; // Altezza fissa per il grafico
        plotContainer.appendChild(plotDiv);

         // Aggiungi il container dopo la tabella di anteprima dati nel #data-visualization-section
        const dataPreviewTableContainer = document.getElementById('data-preview-table'); // Container della tabella
        if(dataPreviewTableContainer) { // Aggiungi dopo il container della tabella
            dataPreviewTableContainer.parentNode.insertBefore(plotContainer, dataPreviewTableContainer.nextSibling);
        } else { // Fallback: aggiungi nel #data-visualization-section se la tabella non si trova
             const dataVisualizationSection = document.getElementById('data-visualization-section');
             if(dataVisualizationSection) {
                dataVisualizationSection.appendChild(plotContainer);
             }
        }
    }

    // Prepara i dati per il grafico 2D
    const trace = {
        x: data.map(row => parseFloat(row[numericFeatures[0]])),
        y: data.map(row => parseFloat(row[numericFeatures[1]])),
        mode: 'markers',
        type: 'scatter',
        marker: { size: 8 },
        text: data.map(row => 
            numericFeatures.map(feature => `${feature}: ${row[feature]}`).join('<br>')
        ),
        hoverinfo: 'text'
    };

    const layout = {
        title: 'Grafico di Dispersione 2D',
        xaxis: { title: numericFeatures[0] },
        yaxis: { title: numericFeatures[1] },
        hovermode: 'closest',
        height: 500
    };

     const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    };

    // Crea il grafico 2D
    Plotly.newPlot('plotly-2d-scatter', [trace], layout, config);

     // Aggiungi controlli per selezionare le feature da visualizzare
    const controlsContainer = document.createElement('div');
    controlsContainer.className = 'mt-4 grid grid-cols-1 md:grid-cols-2 gap-4'; // Due colonne per X e Y
    
    ['X', 'Y'].forEach((axis, index) => {
        const selectContainer = document.createElement('div');
        selectContainer.className = 'space-y-1';
        
        const label = document.createElement('label');
        label.className = 'block text-sm font-medium text-gray-700';
        label.textContent = `Asse ${axis}`;
        
        const select = document.createElement('select');
        select.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
        
        numericFeatures.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = feature;
            // Seleziona le prime 2 feature numeriche per default
            if (index < numericFeatures.length && feature === numericFeatures[index]) {
                 option.selected = true;
            }
            select.appendChild(option);
        });
        // Assicurati che i dropdown abbiano un valore di default se ci sono feature
        if (select.selectedIndex === -1 && numericFeatures.length > 0) {
             select.options[0].selected = true;
        }

        select.addEventListener('change', () => {
            const axisMap = { X: 'x', Y: 'y' };
            const axisKey = axisMap[axis];
            const newData = data.map(row => parseFloat(row[select.value]));
            
            Plotly.restyle('plotly-2d-scatter', {[axisKey]: [newData]}, [0]); // Applica a tutte le tracce
            
            // Aggiorna il titolo dell'asse
            const updatedLayout = {
                [`${axisKey}axis`]: { title: select.value }
            };
            Plotly.relayout('plotly-2d-scatter', updatedLayout);
        });
        
        selectContainer.appendChild(label);
        selectContainer.appendChild(select);
        controlsContainer.appendChild(selectContainer);
    });
    
    plotContainer.appendChild(controlsContainer);
}

// Funzione per creare un grafico di dispersione 3D (rinominata)
function create3dScatterPlot(data, headers) {
    console.log('Chiamata create3dScatterPlot.');
    // Trova le feature numeriche
    const numericFeatures = headers.filter(header => {
        const values = data.map(row => row[header]);
        // Controlla anche che non siano tutte stringhe che non possono essere convertite
        return values.every(val => !isNaN(parseFloat(val)) || (typeof val === 'string' && val.trim() === ''));
    });

     // Rimuovi eventuali feature numeriche che sono solo stringhe vuote
    const validNumericFeatures = numericFeatures.filter(feature => data.some(row => !isNaN(parseFloat(row[feature]))));

    if (validNumericFeatures.length < 3) {
        console.log('Sono necessarie almeno 3 feature numeriche valide per creare un grafico 3D');
         // Rimuovi il grafico precedente se esiste anche se non possiamo crearne uno nuovo
        const existingPlot = document.getElementById('plotly-3d-scatter');
        if (existingPlot) { Plotly.purge('plotly-3d-scatter'); existingPlot.parentNode.removeChild(existingPlot); }

        // Rimuovi anche i controlli degli assi se esistono
        const existingControls = document.querySelector('#3d-scatter-plot-container .grid'); // Seleziona il div con classe grid dentro il container del plot
         if (existingControls) { existingControls.parentNode.removeChild(existingControls); }

         // Rimuovi il container principale del plot 3D
         const existingPlotContainer = document.getElementById('3d-scatter-plot-container');
        if (existingPlotContainer) { existingPlotContainer.parentNode.removeChild(existingPlotContainer); }


        return;
    }

    // Rimuovi il grafico 3D precedente se esiste
    const existingPlot = document.getElementById('plotly-3d-scatter');
    if (existingPlot) {
        Plotly.purge('plotly-3d-scatter');
         // Non rimuovere il parent container (#3d-scatter-plot) perché contiene anche i controlli degli assi
        // existingPlot.parentNode.removeChild(existingPlot);
    }

     // Rimuovi i controlli degli assi precedenti se esistono
    const existingControls = document.querySelector('#3d-scatter-plot-container .grid'); // Seleziona il div con classe grid dentro il container del plot
     if (existingControls) { existingControls.parentNode.removeChild(existingControls); }

    // Crea o trova il container per il grafico 3D
    let plotContainer = document.getElementById('3d-scatter-plot-container');
    if (!plotContainer) {
        plotContainer = document.createElement('div');
        plotContainer.id = '3d-scatter-plot-container';
        plotContainer.className = 'mt-8 p-4 bg-white rounded-lg shadow';
         // Non impostare l'altezza fissa qui, la gestiremo con CSS o Plotly layout
        // plotContainer.style.height = '600px';

        const title = document.createElement('h3');
        title.className = 'text-lg font-medium text-gray-900 mb-4';
        title.textContent = 'Visualizzazione 3D dei Dati';
        plotContainer.appendChild(title);

        // Crea il div per il grafico Plotly
        const plotDiv = document.createElement('div');
        plotDiv.id = 'plotly-3d-scatter';
        plotDiv.style.width = '100%';
        plotDiv.style.height = '500px'; // Altezza fissa
        plotContainer.appendChild(plotDiv);

         // Aggiungi il container dopo la tabella di anteprima dati nel #data-visualization-section
        const dataPreviewTableContainer = document.getElementById('data-preview-table'); // Container della tabella
        if(dataPreviewTableContainer) { // Aggiungi dopo il container della tabella
            dataPreviewTableContainer.parentNode.insertBefore(plotContainer, dataPreviewTableContainer.nextSibling);
        } else { // Fallback: aggiungi nel #data-visualization-section se la tabella non si trova
             const dataVisualizationSection = document.getElementById('data-visualization-section');
             if(dataVisualizationSection) {
                dataVisualizationSection.appendChild(plotContainer);
             }
        }
    } else {
        // Se il container esiste, pulisci solo il contenuto del grafico Plotly
         const plotlyDiv = document.getElementById('plotly-3d-scatter');
        if(plotlyDiv) Plotly.purge(plotlyDiv.id);
    }

    // Prepara i dati per il grafico 3D
    const trace = {
        type: 'scatter3d',
        mode: 'markers',
        x: data.map(row => parseFloat(row[validNumericFeatures[0]])),
        y: data.map(row => parseFloat(row[validNumericFeatures[1]])),
        z: data.map(row => parseFloat(row[validNumericFeatures[2]])),
        marker: {
            size: 5,
            color: data.map((_, i) => i), // Colora i punti in base all'indice
            colorscale: 'Viridis',
            opacity: 0.8
        },
        text: data.map(row => 
            validNumericFeatures.map(feature => `${feature}: ${row[feature]}`).join('<br>')
        ),
        hoverinfo: 'text'
    };

    const layout = {
        title: 'Visualizzazione 3D dei Dati',
        scene: {
            xaxis: { title: validNumericFeatures[0] },
            yaxis: { title: validNumericFeatures.length > 1 ? validNumericFeatures[1] : validNumericFeatures[0] }, // Inizializza Y con la seconda feature se esiste
            zaxis: { title: validNumericFeatures.length > 2 ? validNumericFeatures[2] : validNumericFeatures[0] }  // Inizializza Z con la terza feature se esiste
        },
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 40
        },
        hovermode: 'closest',
        height: 500 // Altezza del layout per il grafico
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
    };

    // Crea il grafico 3D
    Plotly.newPlot('plotly-3d-scatter', [trace], layout, config);

    // Aggiungi controlli per selezionare le feature da visualizzare
    const controlsContainer = document.createElement('div');
    controlsContainer.className = 'mt-4 grid grid-cols-1 md:grid-cols-3 gap-4'; // Rendi i controlli responsivi
    
    ['X', 'Y', 'Z'].forEach((axis, index) => {
        const selectContainer = document.createElement('div');
        selectContainer.className = 'space-y-1';
        
        const label = document.createElement('label');
        label.className = 'block text-sm font-medium text-gray-700';
        label.textContent = `Asse ${axis}`;
        
        const select = document.createElement('select');
        select.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
        
        validNumericFeatures.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = feature;
            // Seleziona le prime 3 feature numeriche per default (o meno se non ce ne sono 3)
            if (index < validNumericFeatures.length && feature === validNumericFeatures[index]) {
                 option.selected = true;
            }
            select.appendChild(option);
        });
         // Se non ci sono 3 feature numeriche, assicurati che i dropdown abbiano un valore di default
        if (select.selectedIndex === -1 && validNumericFeatures.length > 0) {
             select.options[0].selected = true;
        }

        select.addEventListener('change', () => {
            const axisMap = { X: 'x', Y: 'y', Z: 'z' };
            const axisKey = axisMap[axis];
            const newData = data.map(row => parseFloat(row[select.value]));
            
            Plotly.restyle('plotly-3d-scatter', {[axisKey]: [newData]}, [0]); // Applica a tutte le tracce (solo una in questo caso)
            
            // Aggiorna il titolo dell'asse
            const updatedLayout = {
                scene: {
                    [axisKey + 'axis']: { title: select.value }
                }
            };
            Plotly.relayout('plotly-3d-scatter', updatedLayout);
        });
        
        selectContainer.appendChild(label);
        selectContainer.appendChild(select);
        controlsContainer.appendChild(selectContainer);
    });
    
    plotContainer.appendChild(controlsContainer);

    // Aggiungi anche la matrice di correlazione, passandogli solo le feature numeriche valide
    createCorrelationMatrix(data, validNumericFeatures);
}

function createCorrelationMatrix(data, numericFeatures) {
    // Rimuovi la matrice di correlazione precedente se esiste
    const existingMatrix = document.getElementById('correlation-matrix');
    if (existingMatrix) {
        existingMatrix.parentNode.removeChild(existingMatrix);
    }

    // Calcola la matrice di correlazione
    const correlationMatrix = {};
    numericFeatures.forEach(feature1 => {
        correlationMatrix[feature1] = {};
        numericFeatures.forEach(feature2 => {
            const values1 = data.map(row => parseFloat(row[feature1]));
            const values2 = data.map(row => parseFloat(row[feature2]));
            correlationMatrix[feature1][feature2] = calculateCorrelation(values1, values2);
        });
    });

    // Crea il container per la matrice di correlazione
    const matrixContainer = document.createElement('div');
    matrixContainer.id = 'correlation-matrix';
    matrixContainer.className = 'mt-8 p-4 bg-white rounded-lg shadow';

    const title = document.createElement('h3');
    title.className = 'text-lg font-medium text-gray-900 mb-4';
    title.textContent = 'Matrice di Correlazione';
    matrixContainer.appendChild(title);

    // Crea la tabella della matrice di correlazione
    const table = document.createElement('table');
    table.className = 'min-w-full divide-y divide-gray-200';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.appendChild(document.createElement('th')); // Cella vuota per l'angolo
    numericFeatures.forEach(feature => {
        const th = document.createElement('th');
        th.className = 'px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        th.textContent = feature;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    numericFeatures.forEach(feature1 => {
        const row = document.createElement('tr');
        const th = document.createElement('th');
        th.className = 'px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        th.textContent = feature1;
        row.appendChild(th);

        numericFeatures.forEach(feature2 => {
            const td = document.createElement('td');
            td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            const correlation = correlationMatrix[feature1][feature2];
            td.textContent = correlation.toFixed(2);
            // Colora la cella in base al valore di correlazione
            const color = correlation > 0 ?
                `rgba(0, 255, 0, ${Math.abs(correlation)})` :
                correlation < 0 ? `rgba(255, 0, 0, ${Math.abs(correlation)})` : 'rgba(128, 128, 128, 0.5)'; // Gray for 0 correlation
            td.style.backgroundColor = color;
            row.appendChild(td);
        });
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    matrixContainer.appendChild(table);

     // Aggiungi il container dopo il grafico 3D
    const scatterPlotContainer = document.getElementById('3d-scatter-plot-container'); // Corrected ID
     if(scatterPlotContainer) {
         scatterPlotContainer.parentNode.insertBefore(matrixContainer, scatterPlotContainer.nextSibling);
     } else {
         // Fallback if 3d-scatter-plot is not found
         const dataPreviewTable = document.getElementById('data-preview-table');
         if(dataPreviewTable) {
             dataPreviewTable.parentNode.insertBefore(matrixContainer, dataPreviewTable.nextSibling);
         }
     }
}

function calculateCorrelation(x, y) {
    const n = x.length;
    let sum_x = 0;
    let sum_y = 0;
    let sum_xy = 0;
    let sum_x2 = 0;
    let sum_y2 = 0;

    for (let i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    const numerator = n * sum_xy - sum_x * sum_y;
    const denominator = Math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    return denominator === 0 ? 0 : numerator / denominator;
}

// Funzione per popolare il dropdown della colonna target
function populateTargetColumnSelect(columns) {
    const targetColumnSelect = document.getElementById('target-column-select');
    if (!targetColumnSelect) {
        console.error('Elemento select per la colonna target non trovato');
        return;
    }
    
    targetColumnSelect.innerHTML = '<option value="" disabled selected>Scegli colonna...</option>'; // Reset
    
    if (!columns || columns.length === 0) return;

    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        targetColumnSelect.appendChild(option);
    });
    
    console.log('Dropdown colonna target popolato con', columns.length, 'colonne');
}

// Funzione per gestire il caricamento del file CSV
function handleCsvFileUpload(file) {
    console.log('Gestione caricamento file CSV:', file);

    if (!file) {
        console.error('Nessun file selezionato');
        return;
    }

    if (!checkPapaParse()) return;

    // Ottieni i riferimenti alle sezioni
    const analysisSetupSection = document.getElementById('analysis-setup-section');
    const dataVisualizationSection = document.getElementById('data-visualization-section'); // Nuovo contenitore per data e visualizzazioni
    const trainingResultsSection = document.getElementById('training-results-section');
    const inferenceSection = document.getElementById('inference-section');
    const trainModelButton = document.getElementById('train-model-button');

    // --- RESET UI SEZIONI INFERENZA E RISULTATI ALL'INIZIO DEL CARICAMENTO --- 
    // Nascondi le sezioni di analisi, risultati e inferenza inizialmente
    if (analysisSetupSection) analysisSetupSection.classList.add('hidden');
    if (dataVisualizationSection) dataVisualizationSection.classList.add('hidden'); // Nascondi anche visualizzazioni inizialmente
    if (trainingResultsSection) trainingResultsSection.classList.add('hidden');
    if (inferenceSection) inferenceSection.classList.add('hidden');
    if (trainModelButton) trainModelButton.disabled = true;
    
    // Pulisci i contenitori precedenti dei risultati e dell'inferenza
    const metricsContainer = document.getElementById('metrics-container');
    const inferenceFormContainer = document.getElementById('inference-form-container');
    const predictionResultContainer = document.getElementById('prediction-result-container');
    const dataPreviewTable = document.getElementById('data-preview-table'); // Pulisci anche la tabella di anteprima

    if (metricsContainer) metricsContainer.innerHTML = '';
    if (inferenceFormContainer) inferenceFormContainer.innerHTML = '';
    if (predictionResultContainer) predictionResultContainer.innerHTML = '';
    if (dataPreviewTable) dataPreviewTable.innerHTML = '';

    // Rimuovi i grafici precedenti (3D scatter plot e correlation matrix)
    const existingPlot = document.getElementById('plotly-3d-scatter');
    if (existingPlot) { Plotly.purge('plotly-3d-scatter'); existingPlot.parentNode.removeChild(existingPlot); }
    const existingMatrix = document.getElementById('correlation-matrix');
    if (existingMatrix) { existingMatrix.parentNode.removeChild(existingMatrix); }
    // --- FINE RESET UI ---


    try {
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                console.log('Parsing CSV completato con successo');
                console.log('Risultati PapaParse:', results);

                // Verifica la presenza di errori nel parsing
                if (results.errors.length > 0) {
                    console.error('Errori di parsing PapaParse:', results.errors);
                    alert('Errore durante il parsing del CSV: ' + results.errors[0].message);
                    if (analysisSetupSection) analysisSetupSection.classList.add('hidden');
                    if (dataVisualizationSection) dataVisualizationSection.classList.add('hidden');
                    return;
                }

                rawData = results.data;
                headers = results.meta.fields || (rawData.length > 0 ? Object.keys(rawData[0]) : []);

                console.log('Dati grezzi:', rawData);
                console.log('Headers:', headers);

                if (headers.length === 0 && rawData.length > 0) {
                    console.warn('PapaParse non ha estratto gli header, tentando di prenderli dalla prima riga di dati.');
                    // Tentativo di usare la prima riga come headers se non ci sono campi
                     if (rawData.length > 0) {
                        headers = Object.keys(rawData[0]);
                    }
                }

                 // Filtra i dati per rimuovere righe vuote o con tutti i valori mancanti
                rawData = rawData.filter(row => Object.values(row).some(value => value !== null && value !== undefined && value !== ''));
                 console.log('Dati grezzi dopo filtraggio righe vuote:', rawData);

                if (rawData.length > 0 && headers.length > 0) {
                    // Mostra le sezioni appropriate
                     if (dataVisualizationSection) dataVisualizationSection.classList.remove('hidden');
                    if (analysisSetupSection) analysisSetupSection.classList.remove('hidden');

                    displayDataPreview(rawData, headers); // Questa funzione creerà anche i grafici
                    populateTargetColumnSelect(headers);

                    // Simula la selezione della prima colonna target per attivare la logica dipendente
                    const targetColumnSelect = document.getElementById('target-column-select');
                    if (targetColumnSelect && targetColumnSelect.options.length > 1) { // Assicurati che ci siano colonne oltre l'opzione di default
                         // Seleziona la prima colonna (indice 1 perché 0 è l'opzione di default)
                         targetColumnSelect.value = targetColumnSelect.options[1].value;
                        
                         // Attiva manualmente l'evento 'change' per far partire suggestProblemType
                         const event = new Event('change');
                         targetColumnSelect.dispatchEvent(event);
                         console.log('Simulato evento change per colonna target.');
                    } else {
                         // Se non ci sono colonne valide, disabilita il pulsante di training
                         const trainModelButton = document.getElementById('train-model-button');
                         if (trainModelButton) trainModelButton.disabled = true;
                    }

                    // Assicurati che le sezioni di risultati e inferenza siano nascoste
                    if (trainingResultsSection) trainingResultsSection.classList.add('hidden');
                    if (inferenceSection) inferenceSection.classList.add('hidden');


                } else {
                    alert('Errore: Impossibile leggere header o dati validi dal file CSV. Assicurati che il file non sia vuoto e contenga degli header e almeno una riga di dati.');
                    if (analysisSetupSection) analysisSetupSection.classList.add('hidden');
                    if (dataVisualizationSection) dataVisualizationSection.classList.add('hidden');
                }
            },
            error: function(error) {
                console.error('Errore durante il parsing del CSV:', error);
                alert('Errore durante il parsing del CSV: ' + error.message);
                if (analysisSetupSection) analysisSetupSection.classList.add('hidden');
                 if (dataVisualizationSection) dataVisualizationSection.classList.add('hidden');
            }
        });
    } catch (e) {
        console.error('Eccezione durante il parsing del CSV:', e);
        alert('Errore durante l\'elaborazione del file: ' + e.message);
        if (analysisSetupSection) analysisSetupSection.classList.add('hidden');
         if (dataVisualizationSection) dataVisualizationSection.classList.add('hidden');
    }
}

// Funzione per suggerire il tipo di problema in base alla colonna target
function suggestProblemType(targetColumn) {
    console.log('Chiamata suggestProblemType con colonna target:', targetColumn);
    if (!targetColumn || !rawData.length) {
         console.log('suggestProblemType: Colonna target o dati raw mancanti.');
        return;
    }
    
    const problemTypeSelect = document.getElementById('problem-type-select');
    if (!problemTypeSelect) return;
    
    // Analizza i valori della colonna target
    const targetValues = rawData.map(row => row[targetColumn]);
    const uniqueValues = new Set(targetValues);
    
    // Se ci sono pochi valori unici o sono stringhe, suggerisci classificazione
    const isLikelyClassification = uniqueValues.size <= 10 || 
                                 targetValues.some(v => typeof v === 'string');
    
    const determinedProblemType = isLikelyClassification ? 'classification' : 'regression';
    console.log('suggestProblemType: Tipo di problema determinato:', determinedProblemType);
    problemTypeSelect.value = determinedProblemType;
    
    // Popola gli algoritmi disponibili
    populateAlgorithmSelect(problemTypeSelect.value);
    
    // Resetta e aggiorna i parametri dell'algoritmo
    const algorithmSelect = document.getElementById('algorithm-select');
    if (algorithmSelect) {
        algorithmSelect.value = '';
        displayAlgorithmParams(problemTypeSelect.value, '');
    }
    
    checkTrainButtonState();
}

// Funzione per popolare il dropdown degli algoritmi
function populateAlgorithmSelect(problemType) {
    console.log('Chiamata populateAlgorithmSelect con tipo di problema:', problemType); // Log 1
    const algorithmSelect = document.getElementById('algorithm-select');
    if (!algorithmSelect) {
        console.error('populateAlgorithmSelect: Elemento select algoritmo (#algorithm-select) non trovato');
        return;
    }
    console.log('populateAlgorithmSelect: Elemento select algoritmo trovato.');
    
    algorithmSelect.innerHTML = '<option value="" disabled selected>Scegli algoritmo...</option>';
    
    const algorithms = {
        classification: [
            'K-Nearest Neighbors (KNN)',
            'Decision Tree Classifier',
            'Random Forest Classifier'
        ],
        regression: [
            'Linear Regression (OLS)',
            'Polynomial Regression',
            'Support Vector Regression (SVR)',
            'Decision Tree Regressor',
            'Random Forest Regressor'
        ]
    };

    console.log('populateAlgorithmSelect: Oggetto algorithms:', algorithms); // Log 2 - Aggiornato con le liste richieste
    console.log('populateAlgorithmSelect: algorithms[problemType]:', algorithms[problemType]); // Log 3

    const availableAlgorithms = algorithms[problemType] || [];
    console.log('populateAlgorithmSelect: Algoritmi disponibili per', problemType, ':', availableAlgorithms); // Log 4
    
    availableAlgorithms.forEach(algorithm => {
        const option = document.createElement('option');
        option.value = algorithm;
        option.textContent = algorithm;
        algorithmSelect.appendChild(option);
    });

    // Aggiungi un event listener per chiamare checkTrainButtonState quando l'algoritmo cambia
    algorithmSelect.addEventListener('change', function() {
        console.log('Evento change sull\'algoritmo rilevato.');
        displayAlgorithmParams(problemType, this.value); // Già presente, per mostrare i parametri
        checkTrainButtonState(); // Assicurati che lo stato del pulsante venga aggiornato
    });

    // Se ci sono algoritmi disponibili, seleziona il primo e attiva l'evento change
    if (availableAlgorithms.length > 0) {
         algorithmSelect.value = availableAlgorithms[0];
        const event = new Event('change');
        algorithmSelect.dispatchEvent(event);
         console.log('populateAlgorithmSelect: Selezionato il primo algoritmo e simulato evento change.');
    } else {
        // Se non ci sono algoritmi, disabilita il pulsante di training
        checkTrainButtonState();
         console.log('populateAlgorithmSelect: Nessun algoritmo disponibile, chiamo checkTrainButtonState.');
    }
}

// Funzione per mostrare i parametri dell'algoritmo
function displayAlgorithmParams(problemType, algorithmName) {
    const container = document.getElementById('algorithm-params-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    const params = {
        'Logistic Regression': [
            { name: 'C', type: 'float', default: '1.0', min: '0.1', max: '10.0', step: '0.1' }
        ],
        'SVC': [
            { name: 'C', type: 'float', default: '1.0', min: '0.1', max: '10.0', step: '0.1' },
            { name: 'kernel', type: 'select', options: ['linear', 'rbf', 'poly', 'sigmoid'], default: 'rbf' }
        ],
        'Decision Tree Classifier': [
            { name: 'max_depth', type: 'int', default: 'None', min: '1', max: '20', step: '1' },
            { name: 'min_samples_split', type: 'int', default: '2', min: '2', max: '20', step: '1' }
        ],
        'Random Forest Classifier': [
            { name: 'n_estimators', type: 'int', default: '100', min: '10', max: '200', step: '10' },
            { name: 'max_depth', type: 'int', default: 'None', min: '1', max: '20', step: '1' }
        ],
        'Linear Regression': [],
        'Ridge Regression': [
            { name: 'alpha', type: 'float', default: '1.0', min: '0.1', max: '10.0', step: '0.1' }
        ],
        'Lasso Regression': [
            { name: 'alpha', type: 'float', default: '1.0', min: '0.1', max: '10.0', step: '0.1' }
        ],
        'Random Forest Regressor': [
            { name: 'n_estimators', type: 'int', default: '100', min: '10', max: '200', step: '10' },
            { name: 'max_depth', type: 'int', default: 'None', min: '1', max: '20', step: '1' }
        ]
    };
    
    const algorithmParams = params[algorithmName] || [];
    
    algorithmParams.forEach(param => {
        const paramGroup = document.createElement('div');
        paramGroup.className = 'space-y-1';
        
        const label = document.createElement('label');
        label.className = 'block text-sm font-medium text-gray-700';
        label.textContent = param.name;
        
        let input;
        if (param.type === 'select') {
            input = document.createElement('select');
            input.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
            param.options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                if (option === param.default) opt.selected = true;
                input.appendChild(opt);
            });
        } else {
            input = document.createElement('input');
            input.type = param.type === 'float' ? 'number' : 'number';
            input.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
            input.min = param.min;
            input.max = param.max;
            input.step = param.step;
            input.value = param.default;
        }
        
        input.name = `param_${param.name}`;
        input.dataset.paramName = param.name;
        
        paramGroup.appendChild(label);
        paramGroup.appendChild(input);
        container.appendChild(paramGroup);
    });
    
    checkTrainButtonState();
}

// Funzione per verificare se il pulsante di training può essere abilitato
function checkTrainButtonState() {
    console.log('Chiamata checkTrainButtonState.');
    const trainModelButton = document.getElementById('train-model-button');
    if (!trainModelButton) {
        console.error('checkTrainButtonState: Pulsante addestra modello non trovato.');
        return;
    }
    
    const targetColumn = document.getElementById('target-column-select')?.value;
    const problemType = document.getElementById('problem-type-select')?.value;
    const algorithm = document.getElementById('algorithm-select')?.value;

    console.log('checkTrainButtonState: Valori correnti - Colonna Target:', targetColumn, ', Tipo Problema:', problemType, ', Algoritmo:', algorithm);
    
    trainModelButton.disabled = !(targetColumn && problemType && algorithm);
    console.log('checkTrainButtonState: Pulsante disabilitato:', trainModelButton.disabled);
}

// Funzione per gestire il training del modello
async function handleModelTraining() {
    const targetColumn = document.getElementById('target-column-select')?.value;
    const problemType = document.getElementById('problem-type-select')?.value;
    const algorithmName = document.getElementById('algorithm-select')?.value;
    
    if (!targetColumn || !problemType || !algorithmName) {
        alert('Seleziona tutti i campi necessari prima di addestrare il modello');
        return;
    }
    
    // Raccogli i parametri dell'algoritmo
    const algorithmParams = {};
    const paramInputs = document.querySelectorAll('#algorithm-params-container input, #algorithm-params-container select');
    paramInputs.forEach(input => {
        const paramName = input.dataset.paramName;
        if (paramName) {
            let value = input.value;
            if (value === 'None') value = null;
            else if (input.type === 'number') value = Number(value);
            algorithmParams[paramName] = value;
        }
    });
    
    try {
        // Mostra il pulsante come disabilitato durante il training
        const trainModelButton = document.getElementById('train-model-button');
        if (trainModelButton) {
            trainModelButton.disabled = true;
            trainModelButton.textContent = 'Training in corso...';
        }
        
        // Prepara i dati per la richiesta
        const requestData = {
            rawData: rawData,
            targetColumn: targetColumn,
            problemType: problemType,
            algorithmName: algorithmName,
            algorithmParams: algorithmParams
        };
        
        // Invia la richiesta al backend
        const response = await fetch('/train_model_unified', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Errore durante il training del modello');
        }
        
        // Mostra i risultati
        displayTrainingResults(result);
        
    } catch (error) {
        console.error('Errore durante il training:', error);
        alert('Errore durante il training del modello: ' + error.message);
    } finally {
        // Ripristina il pulsante
        const trainModelButton = document.getElementById('train-model-button');
        if (trainModelButton) {
            trainModelButton.disabled = false;
            trainModelButton.textContent = 'Addestra Modello';
        }
    }
}

// Funzione per mostrare i risultati del training
function displayTrainingResults(results) {
    console.log('Risultati del training:', results); // Debug log
    console.log('displayTrainingResults: feature_info received:', results.feature_info); // Log feature_info

    const trainingResultsSection = document.getElementById('training-results-section');
    const metricsContainer = document.getElementById('metrics-container');
    const inferenceSection = document.getElementById('inference-section');
    const inferenceFormContainer = document.getElementById('inference-form-container');

    if (!trainingResultsSection || !metricsContainer || !inferenceSection || !inferenceFormContainer) { // Includi inferenceFormContainer nella verifica
        console.error('Elementi necessari non trovati nel DOM per mostrare i risultati del training.');
        return;
    }

     // --- Pulire il form di inferenza prima di ricrearlo ---
    inferenceFormContainer.innerHTML = '';
    console.log('displayTrainingResults: Contenitore form inferenza pulito.');
     // --- Fine pulizia ---

    // Mostra la sezione dei risultati e inferenza
    trainingResultsSection.classList.remove('hidden');
    inferenceSection.classList.remove('hidden');

    // Mostra le metriche
    let metricsHtml = '<div class="grid grid-cols-2 gap-4">';
    if (results.metrics && typeof results.metrics === 'object') {
        for (const [metric, value] of Object.entries(results.metrics)) {
            metricsHtml += `
                <div class="bg-gray-50 p-3 rounded">
                    <div class="text-sm font-medium text-gray-500">${metric}</div>
                    <div class="mt-1 text-lg font-semibold text-gray-900">${typeof value === 'number' ? value.toFixed(4) : value}</div>
                </div>
            `;
        }
    } else {
        metricsHtml += `
            <div class="col-span-2 bg-gray-50 p-3 rounded">
                <div class="text-sm font-medium text-gray-500">Nessuna metrica disponibile</div>
            </div>
        `;
    }
    metricsHtml += '</div>';
    metricsContainer.innerHTML = metricsHtml;

    // Mostra la sezione di inferenza
    if (results.feature_info) {
        setupInferenceForm(results.feature_info); // Questa funzione popola il form di inferenza
    } else {
        console.warn('Feature_info mancante nei risultati del training. Impossibile impostare il form di inferenza.');
        const inferenceFormContainer = document.getElementById('inference-form-container');
        if(inferenceFormContainer) inferenceFormContainer.innerHTML = "<p>Informazioni sulle feature non disponibili per l'inferenza.</p>";
    }

    // La visualizzazione 3D e la matrice di correlazione sono già state create da displayDataPreview
}

// Funzione per impostare il form di inferenza
function setupInferenceForm(featureInfo) {
    console.log('Chiamata setupInferenceForm.');
    console.log('setupInferenceForm: Using featureInfo:', featureInfo); // Log featureInfo utilizzato
    const container = document.getElementById('inference-form-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    for (const [feature, info] of Object.entries(featureInfo)) {
        const group = document.createElement('div');
        group.className = 'space-y-1';
        
        const label = document.createElement('label');
        label.className = 'block text-sm font-medium text-gray-700';
        label.textContent = feature;
        
        let input;
        if (info.type === 'categorical') {
            // Per valori categorici, creiamo sia un input che un select
            const inputGroup = document.createElement('div');
            inputGroup.className = 'flex space-x-2';
            
            // Input manuale
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
            input.placeholder = 'Inserisci valore manualmente';
            
            // Select per i valori predefiniti
            const select = document.createElement('select');
            select.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
            select.innerHTML = '<option value="">Seleziona un valore...</option>';
            
            info.values.forEach(value => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                select.appendChild(option);
            });
            
            // Quando si seleziona un valore dal select, lo copia nell'input
            select.addEventListener('change', (e) => {
                input.value = e.target.value;
            });
            
            inputGroup.appendChild(input);
            inputGroup.appendChild(select);
            group.appendChild(label);
            group.appendChild(inputGroup);
        } else {
            // Per valori numerici, creiamo un input numerico
            input = document.createElement('input');
            input.type = 'number';
            input.className = 'mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md';
            input.min = info.min;
            input.max = info.max;
            input.value = info.default;
            input.step = (info.max - info.min) / 100;
            
            group.appendChild(label);
            group.appendChild(input);
        }
        
        input.name = `feature_${feature}`;
        input.dataset.featureName = feature;
        
        container.appendChild(group);
    }
    
    // Aggiungi l'event listener per il pulsante di predizione
    const predictButton = document.getElementById('predict-button');
    if (predictButton) {
        predictButton.onclick = handlePrediction;
    }
}

// Funzione per gestire la predizione
async function handlePrediction() {
    const container = document.getElementById('inference-form-container');
    const resultContainer = document.getElementById('prediction-result-container');
    
    if (!container || !resultContainer) return;
    
    // Raccogli i valori delle feature
    const features = {};
    const inputs = container.querySelectorAll('input, select');
    inputs.forEach(input => {
        const featureName = input.dataset.featureName;
        if (featureName) {
            features[featureName] = input.value;
        }
    });
    
    // Debug log: Mostra i dati inviati per la predizione
    console.log('Dati inviati per la predizione:', features);

    try {
        // Mostra un messaggio di caricamento
        resultContainer.innerHTML = '<div class="text-center text-gray-600">Calcolo predizione...</div>';

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Errore durante la predizione');
        }
        
        // Debug log: Mostra il risultato della predizione
        console.log('Risultato predizione ricevuto:', result);

        // Mostra il risultato
        resultContainer.innerHTML = `
            <div class="text-center">
                <div class="text-sm font-medium text-gray-500">Predizione</div>
                <div class="mt-1 text-lg font-semibold text-gray-900">${result.prediction}</div>
            </div>
        `;
        
    } catch (error) {
        console.error('Errore durante la predizione:', error);
        resultContainer.innerHTML = `
            <div class="text-center text-red-600">
                Errore: ${error.message}
            </div>
        `;
    }
}
