// === FRONTEND MACHINE LEARNING PURE CSV FLOW ===
// Dipendenza: PapaParse (CDN inclusa nell'HTML, vedi sotto)
// Flusso: upload CSV -> tabella -> scelta target -> train

// 1. Includi questa riga nel <head> del tuo HTML:
// <script src="https://cdn.jsdelivr.net/npm/papaparse@5.3.2/papaparse.min.js"></script>

// 2. Assicurati che l'HTML abbia questi ID:
// <input type="file" id="csv-upload" />
// <div id="table-section"><table id="csv-table"></table></div>
// <select id="target-select"></select>
// <button id="train-btn" disabled>Train</button>

// Variabili globali
let csvData = [];
let csvColumns = [];
let targetColumn = null;

// === Gestione upload CSV ===
document.addEventListener('DOMContentLoaded', function() {
    const uploadInput = document.getElementById('csv-upload');
    const tableSection = document.getElementById('table-section');
    const table = document.getElementById('csv-table');
    const targetSelect = document.getElementById('target-select');
    const trainBtn = document.getElementById('train-btn');

    // Reset UI
    function resetUI() {
        table.innerHTML = '';
        targetSelect.innerHTML = '<option value="" disabled selected>Scegli la colonna target</option>';
        trainBtn.disabled = true;
        targetColumn = null;
    }

    // Visualizza tabella dati
    function renderTable(data, columns) {
        if (!data.length || !columns.length) return;
        let html = '<thead><tr>' + columns.map(col => `<th>${col}</th>`).join('') + '</tr></thead><tbody>';
        data.forEach(row => {
            html += '<tr>' + columns.map(col => `<td>${row[col] ?? ''}</td>`).join('') + '</tr>';
        });
        html += '</tbody>';
        table.innerHTML = html;
        tableSection.style.display = 'block';
    }

    // Popola menu a tendina
    function populateDropdown(columns) {
        targetSelect.innerHTML = '<option value="" disabled selected>Scegli la colonna target</option>';
        columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            targetSelect.appendChild(opt);
        });
    }

    // Gestione upload
    uploadInput.addEventListener('change', function(e) {
        resetUI();
        const file = e.target.files[0];
        if (!file) return;
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                csvData = results.data;
                csvColumns = results.meta.fields;
                renderTable(csvData, csvColumns);
                populateDropdown(csvColumns);
            },
            error: function(err) {
                alert('Errore parsing CSV: ' + err.message);
            }
        });
    });

    // Gestione scelta target
    targetSelect.addEventListener('change', function() {
        targetColumn = this.value;
        trainBtn.disabled = !targetColumn;
    });

    // Gestione click Train
    trainBtn.addEventListener('click', function() {
        if (!targetColumn) {
            alert('Seleziona prima la colonna target!');
            return;
        }
        // Qui puoi inviare csvData e targetColumn al backend per il training
        alert('Training avviato su target: ' + targetColumn + '\nRighe: ' + csvData.length);
        // Esempio: fetch('/train', { method: 'POST', body: JSON.stringify({data: csvData, target: targetColumn}), headers: {'Content-Type': 'application/json'}})
    });

    // Inizializza UI
    resetUI();
});
// === FINE LOGICA MINIMALE ===


let problemType = null;
let trainedModel = null;
let scatterPlot = null;
let fitCurve = null;
let datasetData = null;

// Funzioni di utilità
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Funzione per mostrare messaggi di errore
function showError(message) {
    console.error(message);
    showToast(message, 'error');
}

// Funzione per mostrare messaggi di successo
function showSuccess(message) {
    console.log(message);
    showToast(message, 'success');
}

// Funzione per mostrare notifiche toast
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    
    // Crea il container delle notifiche se non esiste
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed bottom-4 right-4 z-50 flex flex-col gap-2';
        document.body.appendChild(container);
    }
    
    // Crea la notifica toast
    const toast = document.createElement('div');
    toast.className = 'p-4 rounded-lg shadow-lg flex items-center justify-between min-w-[300px] animate-fade-in';
    
    // Imposta lo stile in base al tipo
    if (type === 'error') {
        toast.classList.add('bg-red-500', 'text-white');
    } else if (type === 'success') {
        toast.classList.add('bg-green-500', 'text-white');
    } else {
        toast.classList.add('bg-blue-500', 'text-white');
    }
    
    // Aggiungi il messaggio
    const messageEl = document.createElement('span');
    messageEl.textContent = message;
    toast.appendChild(messageEl);
    
    // Aggiungi il pulsante di chiusura
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '&times;';
    closeBtn.className = 'ml-4 text-white font-bold text-xl';
    closeBtn.onclick = function() {
        toast.remove();
    };
    toast.appendChild(closeBtn);
    
    // Aggiungi la notifica al container
    document.getElementById('toast-container').appendChild(toast);
    
    // Rimuovi la notifica dopo 5 secondi
    setTimeout(() => {
        if (toast.parentNode) {
            toast.classList.add('animate-fade-out');
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

// Funzione per inizializzare l'interfaccia di inferenza
function initInference(featureInfo) {
    console.log('Inizializzazione interfaccia inferenza con:', featureInfo);
    
    const predictionInputs = document.getElementById('prediction-inputs');
    const predictBtn = document.getElementById('predict-btn');
    const predictionResult = document.getElementById('prediction-result');
    const predictionValue = document.getElementById('prediction-value');
    
    if (!predictionInputs || !predictBtn) {
        console.error('Elementi per inferenza non trovati');
        showError('Elementi per inferenza non trovati nella pagina');
        return;
    }
    
    // Rimuovi eventuali listener precedenti
    const oldBtn = predictBtn.cloneNode(true);
    predictBtn.parentNode.replaceChild(oldBtn, predictBtn);
    
    // Svuota il container degli input
    predictionInputs.innerHTML = '';
    
    // Nascondi il risultato precedente se presente
    if (predictionResult) {
        predictionResult.classList.add('hidden');
    }
    
    // Crea gli input per ogni feature
    for (const [feature, info] of Object.entries(featureInfo)) {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'flex flex-col';
        
        const label = document.createElement('label');
        label.className = 'text-sm font-medium text-gray-700 mb-1';
        label.textContent = feature;
        
        let input;
        
        if (info.type === 'numeric') {
            input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.min = info.min;
            input.max = info.max;
            input.value = info.default;
            input.className = 'p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2c2063]';
        } else {
            input = document.createElement('select');
            input.className = 'p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2c2063]';
            
            // Aggiungi le opzioni
            for (const value of info.values) {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                input.appendChild(option);
            }
        }
        
        input.id = `feature-${feature}`;
        input.dataset.feature = feature;
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        predictionInputs.appendChild(inputGroup);
    }
    
    // Ottieni il nuovo riferimento al pulsante dopo la sostituzione
    const newPredictBtn = document.getElementById('predict-btn');
    
    // Gestisci il click sul pulsante di predizione
    newPredictBtn.addEventListener('click', function(event) {
        // Previeni comportamenti di default
        event.preventDefault();
        event.stopPropagation();
        
        console.log('Click sul pulsante PREDICT rilevato');
        
        // Raccogli i valori degli input
        const featureValues = {};
        const inputs = predictionInputs.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            const feature = input.dataset.feature;
            let value = input.value;
            
            // Converti in numero se è un input numerico
            if (input.type === 'number') {
                value = parseFloat(value);
            }
            
            featureValues[feature] = value;
        });
        
        console.log('Valori delle feature per la predizione:', featureValues);
        
        // Mostra loader
        newPredictBtn.disabled = true;
        newPredictBtn.innerHTML = '<span class="inline-block h-5 w-5 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></span> Predizione in corso...';
        
        // Invia richiesta al backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                features: featureValues
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Errore HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Risposta predizione:', data);
            if (data.success) {
                // Formatta il valore della predizione in base al tipo di problema
                let formattedPrediction = data.prediction;
                
                // Se è un numero, formatta con 4 decimali
                if (typeof data.prediction === 'number') {
                    formattedPrediction = data.prediction.toFixed(4);
                }
                
                // Mostra il risultato
                predictionValue.textContent = formattedPrediction;
                predictionResult.classList.remove('hidden');
                
                // Aggiungi una classe per animare il risultato
                predictionResult.classList.add('animate-fade-in');
                
                showSuccess('Predizione completata con successo!');
            } else {
                showError('Errore nella predizione: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Errore durante la predizione:', error);
            showError(error.message || 'Errore durante la predizione');
        })
        .finally(() => {
            // Ripristina il pulsante
            newPredictBtn.disabled = false;
            newPredictBtn.innerHTML = 'Predici';
        });
    });
    
    // Mostra la sezione di inferenza
    const inferenceSection = document.getElementById('inference-section');
    if (inferenceSection) {
        inferenceSection.classList.remove('hidden');
        // Scorri fino alla sezione di inferenza
        setTimeout(() => {
            inferenceSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }
}

// Inizializzazione al caricamento del DOM
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded - Machine Learning JS v2');
    
    // Elementi DOM
    const fileUploadInput = document.getElementById('dataset-upload'); // Renamed for clarity
    const dropZone = document.getElementById('drop-zone');
    const browseBtn = document.getElementById('browse-btn');
    // const analyzeBtn = document.getElementById('analyze-btn'); // Removed
    const fileInfoBox = document.getElementById('file-info'); // Renamed for clarity
    const fileNameDisplay = document.getElementById('file-name'); // Renamed for clarity
    const fileSizeDisplay = document.getElementById('file-size'); // Renamed for clarity
    // const analysisSection = document.getElementById('analysis-section'); // Not directly used here anymore
    const targetColumnSelect = document.getElementById('target-column');
    const analyzeTargetBtn = document.getElementById('analyze-target-btn');
    const problemTypeSelect = document.getElementById('problem-type');
    const trainBtn = document.getElementById('train-btn');
    
    // Debug - Verifica che gli elementi DOM siano stati trovati
    console.log('File upload input element:', fileUploadInput);
    console.log('Drop zone element:', dropZone);
    console.log('Browse button element:', browseBtn);
    // console.log('Analyze button element:', analyzeBtn); // Removed
    
    // Inizializzazione UI
    if (fileInfoBox) {
        fileInfoBox.classList.add('hidden');
    }
    
    // Gestione tutorial
    initTutorial(); // Assuming initTutorial() handles its own element checks
    
    // Gestione caricamento file e drag & drop
    if (fileUploadInput && dropZone && browseBtn && fileNameDisplay && fileSizeDisplay && fileInfoBox) {
        // Pass fileUploadInput to initFileUpload as it's the actual input element
        initFileUpload(fileUploadInput, dropZone, browseBtn, fileInfoBox, fileNameDisplay, fileSizeDisplay);
        // Pass fileUploadInput to setupDragAndDrop so it can use it if needed, or pass to handleFileSelection
        setupDragAndDrop(dropZone, fileUploadInput, fileNameDisplay, fileSizeDisplay, fileInfoBox);
    } else {
        console.error('One or more elements for file upload/drag-drop not found. Upload/drop will not work.');
        showError('Impossibile inizializzare il caricamento file. Elementi UI mancanti.');
    }
    
    // Gestione analisi dataset - Call removed: initDatasetAnalysis(analyzeBtn, analysisSection, targetColumnSelect);
    
    // Gestione analisi colonna target
    if (analyzeTargetBtn && targetColumnSelect) {
        initTargetAnalysis(analyzeTargetBtn, targetColumnSelect);
    } else {
        console.warn('Elements for target analysis not found. Target analysis UI might not work.');
    }
    
    // Gestione training modello
    if (trainBtn && problemTypeSelect) {
        initModelTraining(trainBtn, problemTypeSelect);
    } else {
        console.warn('Elements for model training not found. Model training UI might not work.');
    }
});

// Inizializzazione tutorial
function initTutorial() {
    const tutorialBtn = document.getElementById('tutorial-btn');
    const tutorialModal = document.getElementById('tutorial-modal');
    const closeTutorialBtn = document.getElementById('close-tutorial-btn');
    
    if (tutorialBtn && tutorialModal && closeTutorialBtn) {
        tutorialBtn.addEventListener('click', function() {
            tutorialModal.classList.remove('hidden');
        });
        
        closeTutorialBtn.addEventListener('click', function() {
            tutorialModal.classList.add('hidden');
        });
        
        // Chiudi il modal se si clicca fuori
        tutorialModal.addEventListener('click', function(e) {
            if (e.target === tutorialModal) {
                tutorialModal.classList.add('hidden');
            }
        });
    }
}

// Inizializzazione caricamento file
function initFileUpload(fileUploadInput, dropZone, browseBtn, fileInfoBox, fileNameDisplay, fileSizeDisplay) {
    console.log('Attempting to initFileUpload. Elements received:', { fileUploadInput, browseBtn, dropZone });
    if (!fileUploadInput || !browseBtn) { // dropZone is not strictly essential for browse functionality here, but good to check if passed
        console.error('Essential elements (fileUploadInput or browseBtn) not found or null in initFileUpload.');
        console.error('Elementi DOM per il caricamento file non trovati');
        return;
    }
    
    console.log('Inizializzazione caricamento file...');
    
    // Gestione input file
    fileUploadInput.addEventListener('change', function(e) {
        console.log('File input changed');
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0], fileNameDisplay, fileSizeDisplay, fileInfoBox);
        }
    });
    
    // Gestione click sul pulsante "sfoglia"
    browseBtn.addEventListener('click', function() {
        console.log('Browse button clicked');
        fileUploadInput.click();
    });
    
    // Drag and drop is initialized in DOMContentLoaded
}

// Gestione selezione file
function handleFileSelection(file, fileNameDisplay, fileSizeDisplay, fileInfoBox) {
    console.log('File selezionato:', file.name, file.type);
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showError('Il file deve essere in formato CSV');
        return;
    }
    
    window.uploadedFile = file;
    console.log('File salvato globalmente:', window.uploadedFile.name);
    
    // Mostra info file
    fileNameDisplay.textContent = window.uploadedFile.name;
    fileSizeDisplay.textContent = formatFileSize(window.uploadedFile.size);
    fileInfoBox.classList.remove('hidden');
    
    // Rendi visibile la sezione di raccomandazione
    const recommendationSection = document.getElementById('recommendation-section');
    if (recommendationSection && recommendationSection.classList.contains('hidden')) {
        recommendationSection.classList.remove('hidden');
    }
    
    // Rendi visibile la sezione di training
    const trainingSection = document.getElementById('training-section');
    if (trainingSection && trainingSection.classList.contains('hidden')) {
        trainingSection.classList.remove('hidden');
    }
    
    // Carica automaticamente le colonne del dataset
    const formData = new FormData();
    formData.append('file', window.uploadedFile);
    
    console.log('Caricamento automatico colonne dataset:', window.uploadedFile.name);
    showToast('Caricamento colonne dataset in corso...', 'info');
    
    // Definiamo alcune colonne di default in caso di errore
    // Questo garantisce che ci siano sempre colonne disponibili
    window.datasetColumns = ['colonna1', 'colonna2', 'target'];
    console.log('Colonne di default impostate:', window.datasetColumns);
    
    fetch('/analyze_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Errore HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Risposta analisi automatica completa:', data);
        if (data.success && data.columns && data.columns.length > 0) {
            // Salva le colonne del dataset come variabile globale
            window.datasetColumns = data.columns;
            console.log('Colonne dataset caricate con successo:', window.datasetColumns);
            
            // Salva i dati del dataset
            window.datasetData = data.data;

            // Mostra la tabella del dataset e popola il select della colonna target
            if (window.datasetData && window.datasetColumns) {
                displayDatasetTable(window.datasetData, window.datasetColumns);
                populateTargetColumnSelect(window.datasetColumns);
            } else {
                console.error('Dati del dataset o colonne mancanti dopo analisi.');
                showError('Errore nel caricamento dei dati del dataset.');
            }

            // Disabilita il pulsante di training finché non viene selezionata una colonna target valida
            const trainBtn = document.getElementById('train-btn');
            if (trainBtn) {
                trainBtn.disabled = true;
                trainBtn.classList.remove('animate-pulse');
            }
        
            
            // Scorri fino alla tabella del dataset
            setTimeout(() => {
                const datasetSection = document.getElementById('dataset-table-section'); // Define datasetSection here
                if (datasetSection) {
                    datasetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    console.warn('Dataset section (dataset-table-section) not found for scrolling.');
                }
            }, 300);
            
            showSuccess('Dataset caricato con successo con ' + window.datasetColumns.length + ' colonne');
        } else {
            console.warn('Risposta API valida ma senza colonne, usando colonne di default');
            showToast('Usando colonne di default per il dataset', 'warning');
        }
    })
    .catch(error => {
        console.error('Errore durante l\'analisi automatica del dataset:', error);
        showToast('Usando colonne di default per il dataset', 'warning');
    });
    
    showSuccess('File caricato con successo: ' + window.uploadedFile.name);
}

// Setup drag and drop
function setupDragAndDrop(dropZone, fileUploadInput, fileNameDisplay, fileSizeDisplay, fileInfoBox) {
    console.log('Attempting to setupDragAndDrop. Elements received:', { dropZone, fileUploadInput });
    if (!dropZone) {
        console.error('Essential element (dropZone) not found or null in setupDragAndDrop.');
        return;
    }
    // Previeni il comportamento di default del browser per i file trascinati
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Evidenzia la zona di drop quando si trascina un file sopra di essa
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, function() {
            console.log('Drag over drop zone');
            dropZone.querySelector('div').classList.add('border-[#2c2063]', 'bg-blue-50');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, function() {
            dropZone.querySelector('div').classList.remove('border-[#2c2063]', 'bg-blue-50');
        }, false);
    });
    
    // Gestisci il file rilasciato
    dropZone.addEventListener('drop', function(e) {
        console.log('File dropped');
        const dt = e.dataTransfer;
        if (dt.files.length > 0) {
            handleFileSelection(dt.files[0], fileNameDisplay, fileSizeDisplay, fileInfoBox);
        }
    }, false);
}


// Inizializzazione analisi colonna target
function initTargetAnalysis(analyzeTargetBtn, targetColumnSelect) {
    if (!analyzeTargetBtn || !targetColumnSelect) {
        console.error('Elementi per analisi target non trovati');
        showError('Elementi per analisi target non trovati nella pagina');
        return;
    }
    
    // Rimuovi eventuali listener precedenti
    const oldBtn = analyzeTargetBtn.cloneNode(true);
    analyzeTargetBtn.parentNode.replaceChild(oldBtn, analyzeTargetBtn);
    analyzeTargetBtn = oldBtn;
    
    targetColumnSelect.addEventListener('change', function() {
        console.log('Colonna target selezionata:', this.value);
        analyzeTargetBtn.disabled = false;
    });
    
    analyzeTargetBtn.addEventListener('click', function() {
        targetColumn = targetColumnSelect.value;
        if (!targetColumn) {
            showError('Seleziona una colonna target');
            return;
        }
        
        // Mostra loader
        analyzeTargetBtn.disabled = true;
        analyzeTargetBtn.innerHTML = '<span class="inline-block h-5 w-5 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></span> Analisi in corso...';
        
        console.log('Invio richiesta di analisi per la colonna target:', targetColumn);
        
        // Invia richiesta al backend
        fetch('/analyze_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_column: targetColumn
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Errore HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Risposta analisi target:', data);
            if (data.success) {
                // Aggiorna l'interfaccia con i risultati dell'analisi
                document.getElementById('target-analysis').innerHTML = data.analysis;
                document.getElementById('target-analysis-section').classList.remove('hidden');
                
                // Aggiorna il tipo di problema suggerito
                if (data.problem_type) {
                    const problemTypeSelect = document.getElementById('problem-type');
                    problemTypeSelect.value = data.problem_type;
                    problemType = data.problem_type;
                }
                
                // Abilita il pulsante di training
                const trainBtn = document.getElementById('train-btn');
                if (trainBtn) {
                    console.log('Abilitazione pulsante TRAIN');
                    trainBtn.disabled = false;
                    // Evidenzia il pulsante per attirare l'attenzione
                    trainBtn.classList.add('animate-pulse');
                    setTimeout(() => {
                        trainBtn.classList.remove('animate-pulse');
                    }, 2000);
                    
                    // Scorri fino al pulsante di training
                    setTimeout(() => {
                        trainBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 300);
                } else {
                    console.error('Pulsante TRAIN non trovato');
                }
                
                showSuccess('Analisi della colonna target completata');
            } else {
                showError('Errore nell\'analisi della colonna target: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Errore durante l\'analisi della colonna target:', error);
            showError(error.message || 'Errore durante l\'analisi della colonna target');
        })
        .finally(() => {
            // Ripristina il pulsante
            analyzeTargetBtn.disabled = false;
            analyzeTargetBtn.innerHTML = 'Analizza';
        });
    });
}

// Inizializzazione training modello
// Funzione per aggiornare i grafici con i dati ricevuti dal backend
function updatePlots(plotData) {
    try {
        console.log('Aggiornamento grafici con dati:', plotData);
        
        // Aggiorna il grafico di dispersione
        if (plotData.scatter && document.getElementById('scatter-plot')) {
            const scatterCtx = document.getElementById('scatter-plot').getContext('2d');
            
            // Distruggi il grafico esistente se presente
            if (scatterPlot) {
                scatterPlot.destroy();
            }
            
            // Crea il nuovo grafico
            scatterPlot = new Chart(scatterCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Dati di test',
                        data: plotData.scatter.points,
                        backgroundColor: 'rgba(255, 56, 92, 0.7)',
                        borderColor: 'rgba(255, 56, 92, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: plotData.scatter.x_label || 'Valore reale'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: plotData.scatter.y_label || 'Predizione'
                            }
                        }
                    }
                }
            });
        }
        
        // Aggiorna la curva di fit per regressione
        if (plotData.fit_curve && document.getElementById('fit-curve')) {
            const fitCtx = document.getElementById('fit-curve').getContext('2d');
            
            // Distruggi il grafico esistente se presente
            if (fitCurve) {
                fitCurve.destroy();
            }
            
            // Crea il nuovo grafico
            fitCurve = new Chart(fitCtx, {
                type: 'line',
                data: {
                    labels: plotData.fit_curve.x,
                    datasets: [
                        {
                            label: 'Dati reali',
                            data: plotData.fit_curve.y_true,
                            backgroundColor: 'rgba(53, 162, 235, 0.5)',
                            borderColor: 'rgba(53, 162, 235, 1)',
                            pointRadius: 3,
                            pointHoverRadius: 5
                        },
                        {
                            label: 'Curva di fit',
                            data: plotData.fit_curve.y_pred,
                            backgroundColor: 'rgba(255, 56, 92, 0.5)',
                            borderColor: 'rgba(255, 56, 92, 1)',
                            borderWidth: 2,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: plotData.fit_curve.x_label || 'Feature'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: plotData.fit_curve.y_label || 'Target'
                            }
                        }
                    }
                }
            });
        }
        
        // Mostra la sezione di visualizzazione
        document.getElementById('visualization-section').classList.remove('hidden');
        
    } catch (error) {
        console.error('Errore nell\'aggiornamento dei grafici:', error);
    }
}

// Funzione per aggiornare le metriche del modello
function updateMetrics(metrics) {
    try {
        console.log('Aggiornamento metriche:', metrics);
        const metricsContainer = document.getElementById('model-metrics');
        
        if (!metricsContainer) {
            console.error('Container delle metriche non trovato');
            return;
        }
        
        // Svuota il container
        metricsContainer.innerHTML = '';
        
        // Aggiungi le metriche al container
        for (const [key, value] of Object.entries(metrics)) {
            const metricCard = document.createElement('div');
            metricCard.className = 'bg-white rounded-lg p-3 shadow-sm';
            
            const metricName = document.createElement('div');
            metricName.className = 'text-sm font-medium text-gray-500 mb-1';
            metricName.textContent = key.replace(/_/g, ' ').toUpperCase();
            
            const metricValue = document.createElement('div');
            metricValue.className = 'text-xl font-bold text-[#2c2063]';
            metricValue.textContent = typeof value === 'number' ? value.toFixed(4) : value;
            
            metricCard.appendChild(metricName);
            metricCard.appendChild(metricValue);
            metricsContainer.appendChild(metricCard);
        }
    } catch (error) {
        console.error('Errore nell\'aggiornamento delle metriche:', error);
    }
}

// Funzione per abilitare/disabilitare il pulsante di TRAIN
function checkTrainButtonState() {
    const trainBtn = document.getElementById('train-btn');
    if (!trainBtn) {
        console.error('Pulsante TRAIN (train-btn) non trovato in checkTrainButtonState.');
        return;
    }

    // Controlla se tutti i prerequisiti sono soddisfatti
    if (window.uploadedFile && window.targetColumn && window.problemType) {
        trainBtn.disabled = false;
        console.log('Pulsante TRAIN abilitato.');
    } else {
        trainBtn.disabled = true;
        console.log('Pulsante TRAIN disabilitato. Prerequisiti: uploadedFile, targetColumn, problemType. Stati correnti -> File caricato:', !!window.uploadedFile, 'Colonna target:', window.targetColumn, 'Tipo problema:', window.problemType);
    }
}

// Funzione per visualizzare il dataset in una tabella
function displayDatasetTable(data, columns) {
    console.log('Visualizzazione dataset in tabella:', data.length, 'righe,', columns.length, 'colonne');
    
    const headerRow = document.getElementById('dataset-header');
    const tableBody = document.getElementById('dataset-body');
    
    if (!headerRow || !tableBody) {
        console.error('Elementi della tabella non trovati');
        return;
    }
    
    // Pulisci la tabella
    headerRow.innerHTML = '';
    tableBody.innerHTML = '';
    
    // Aggiungi le intestazioni delle colonne
    columns.forEach(column => {
        const th = document.createElement('th');
        th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        th.textContent = column;
        headerRow.appendChild(th);
    });
    
    // Aggiungi le righe di dati (limita a 100 righe per prestazioni)
    const maxRows = Math.min(data.length, 100);
    for (let i = 0; i < maxRows; i++) {
        const row = document.createElement('tr');
        row.className = i % 2 === 0 ? 'bg-white' : 'bg-gray-50';
        
        columns.forEach(column => {
            const td = document.createElement('td');
            td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
            td.textContent = data[i][column] !== undefined ? data[i][column] : 'N/A';
            row.appendChild(td);
        });
        
        tableBody.appendChild(row);
    }
    
    // Se ci sono più di 100 righe, mostra un messaggio
    if (data.length > 100) {
        const infoRow = document.createElement('tr');
        const infoCell = document.createElement('td');
        infoCell.colSpan = columns.length;
        infoCell.className = 'px-6 py-4 text-center text-sm text-gray-500 bg-gray-100';
        infoCell.textContent = `Mostrate le prime 100 righe di ${data.length} totali`;
        infoRow.appendChild(infoCell);
        tableBody.appendChild(infoRow);
    }
}

// Funzione per popolare il dropdown della colonna target
function populateTargetColumnSelect(columns) {
    const targetColumnSelect = document.getElementById('target-column');
    if (!targetColumnSelect) {
        console.error('Elemento select della colonna target (target-column) non trovato.');
        return;
    }

    targetColumnSelect.innerHTML = '<option value="" disabled selected>Seleziona una colonna</option>'; // Placeholder

    if (columns && columns.length > 0) {
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            targetColumnSelect.appendChild(option);
        });
        if (window.targetColumn && columns.includes(window.targetColumn)) {
            targetColumnSelect.value = window.targetColumn;
        } else {
             window.targetColumn = null;
        }
    } else {
        window.targetColumn = null;
    }
    
    // Rimuovi e riaggiungi l'event listener per evitare duplicazioni e assicurare che sia sull'elemento corretto
    const newSelect = targetColumnSelect.cloneNode(true);
    targetColumnSelect.parentNode.replaceChild(newSelect, targetColumnSelect);
    
    newSelect.addEventListener('change', function() {
        window.targetColumn = newSelect.value;
        if (window.targetColumn) { // Solo se una colonna valida è selezionata
            showToast(`Colonna target selezionata: ${window.targetColumn}`, 'info');
            // Abilita il pulsante di training
            const trainBtn = document.getElementById('train-btn');
            if (trainBtn) {
                trainBtn.disabled = false;
                trainBtn.classList.add('animate-pulse');
                setTimeout(() => {
                    trainBtn.classList.remove('animate-pulse');
                }, 2000);
            }
        } else {
            // Disabilita il pulsante di training se la selezione non è valida
            const trainBtn = document.getElementById('train-btn');
            if (trainBtn) trainBtn.disabled = true;
        }
        console.log('Colonna target cambiata a:', window.targetColumn);
        checkTrainButtonState(); // Verifica lo stato del pulsante TRAIN
    });

    checkTrainButtonState(); // Chiamata iniziale per impostare lo stato del pulsante
}

// Funzione per generare lo scatter plot
function generateScatterPlot(scatterData, targetColumnName) {
    console.log('Generazione scatter plot con dati:', scatterData, 'Target:', targetColumnName);
    const canvas = document.getElementById('scatter-plot');
    if (!canvas) {
        console.error('Canvas per scatter plot (scatter-plot) non trovato.');
        return;
    }
    if (window.scatterPlot) {
        window.scatterPlot.destroy();
        window.scatterPlot = null;
    }

    const xValues = scatterData.x_values || []; 
    const yValues = scatterData.y_values || [];
    const predictions = scatterData.predictions || [];

    if (xValues.length === 0 || yValues.length === 0) {
        console.warn('Dati per scatter plot (x_values o y_values) mancanti o vuoti.');
        const scatterPlotSection = document.getElementById('scatter-plot-section');
        if (scatterPlotSection) scatterPlotSection.classList.add('hidden');
        showError('Dati insufficienti per generare lo scatter plot.');
        return;
    }

    const realDataset = {
        label: 'Dati Reali',
        data: xValues.map((x, i) => ({ x: x, y: yValues[i] })),
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
        borderColor: 'rgba(54, 162, 235, 1)',
        pointRadius: 5,
        pointHoverRadius: 7,
        type: 'scatter'
    };

    const datasets = [realDataset];

    if (predictions.length === xValues.length) {
        const predictionDataset = {
            label: 'Predizioni Modello',
            data: xValues.map((x, i) => ({ x: x, y: predictions[i] })),
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.1,
            type: 'line'
        };
        datasets.push(predictionDataset);
    } else if (predictions.length > 0) {
        console.warn('Lunghezza delle predizioni non corrisponde ai dati reali. Le predizioni non verranno mostrate sulla linea.');
    }

    window.scatterPlot = new Chart(canvas, {
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: scatterData.x_label || 'Feature (o Indice del campione)' 
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: targetColumnName || 'Valore Target'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const datasetLabel = context.dataset.label || '';
                            let label = datasetLabel;
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(4);
                            }
                            return label;
                        }
                    }
                },
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: `Predizioni vs Valori Reali per ${targetColumnName}`
                }
            }
        }
    });
    const scatterPlotSection = document.getElementById('scatter-plot-section');
    if (scatterPlotSection) {
        scatterPlotSection.classList.remove('hidden');
        scatterPlotSection.scrollIntoView({ behavior: 'smooth' });
    }
    console.log('Scatter plot generato/aggiornato con successo.');
}

// Funzione per creare il form di inferenza con i campi dinamici
function createInferenceForm(columns, targetColumn, featureInfo) {
    console.log('Creazione form di inferenza per colonne:', columns, 'Target:', targetColumn, 'Info:', featureInfo);
    
    const inferenceInputs = document.getElementById('inference-inputs');
    const inferenceForm = document.getElementById('inference-form');
    
    if (!inferenceInputs || !inferenceForm) {
        console.error('Elementi per inferenza (inference-inputs, inference-form) non trovati.');
        return;
    }
    
    inferenceInputs.innerHTML = ''; 
    
    const notificationText = document.querySelector('#inference-section .bg-yellow-100 p');
    if (notificationText) {
        notificationText.textContent = `Compila i campi per ottenere una predizione. Il modello predice per '${targetColumn}'.`;
    }
    
    if (!Array.isArray(columns)) {
        console.error('Errore in createInferenceForm: "columns" non è un array.', columns);
        showError('Errore interno: impossibile generare il form di inferenza.');
        return;
    }

    columns.forEach(column => {
        if (column !== targetColumn) {
            const formGroup = document.createElement('div');
            formGroup.className = 'mb-4';
            
            const label = document.createElement('label');
            label.htmlFor = `feature-${column}`;
            label.className = 'block text-sm font-medium text-gray-700 mb-1';
            label.textContent = column;
            
            const input = document.createElement('input');
            input.id = `feature-${column}`;
            input.name = column; 
            input.className = 'w-full p-2 border border-gray-300 rounded-md focus:ring-[#2c2063] focus:border-[#2c2063]';
            input.required = true;

            if (featureInfo && featureInfo[column]) {
                const info = featureInfo[column];
                if (info.type === 'numeric' || typeof info.default === 'number' || typeof info.mean === 'number') {
                    input.type = 'number';
                    input.step = 'any'; 
                    if (info.min !== undefined) input.min = info.min;
                    if (info.max !== undefined) input.max = info.max;
                    input.placeholder = info.mean !== undefined ? `Es. (Media): ${info.mean.toFixed(2)}` : (info.default !== undefined ? `Es: ${info.default}` : 'Valore numerico');
                } else if (info.type === 'categorical' && info.values && info.values.length > 0) {
                    input.type = 'text';
                    input.placeholder = `Es: ${info.values[0]}`;
                } else {
                    input.type = 'text';
                    input.placeholder = 'Valore';
                }
            } else {
                input.type = 'text'; 
                input.placeholder = 'Valore';
            }
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
            inferenceInputs.appendChild(formGroup);
        }
    });

    const newInferenceForm = inferenceForm.cloneNode(true); 
    inferenceForm.parentNode.replaceChild(newInferenceForm, inferenceForm);
    
    newInferenceForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const featureValues = {};
        let allValid = true;
        columns.forEach(col => {
            if (col !== targetColumn) {
                const inputElement = newInferenceForm.querySelector(`#feature-${col}`);
                if (inputElement) {
                    let value = inputElement.value;
                    if (inputElement.type === 'number') {
                        value = parseFloat(value);
                        if (isNaN(value)) {
                            showError(`Valore non valido per ${col}. Inserire un numero.`);
                            allValid = false;
                        }
                    }
                    featureValues[col] = value;
                } else {
                    console.warn(`Input per la feature ${col} non trovato nel form di inferenza.`);
                }
            }
        });

        if (!allValid) return;
        
        console.log('Valori feature per inferenza:', featureValues);
        
        const payload = {
            features: featureValues,
        };
        
        const submitBtn = newInferenceForm.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent mr-2"></span> Predizione in corso...';
        
        const predictionResultEl = document.getElementById('prediction-result');
        const predictionValueEl = document.getElementById('prediction-value');

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || `Errore HTTP: ${response.status}`); });
            }
            return response.json();
        })
        .then(data => {
            console.log('Risposta predizione:', data);
            if (data.success) {
                if (predictionResultEl && predictionValueEl) {
                    predictionValueEl.textContent = typeof data.prediction === 'number' ? data.prediction.toFixed(4) : data.prediction;
                    predictionResultEl.classList.remove('hidden');
                    predictionValueEl.classList.add('animate-pulse');
                    setTimeout(() => predictionValueEl.classList.remove('animate-pulse'), 2000);
                }
                showSuccess('Predizione completata con successo!');
            } else {
                showError('Errore nella predizione: ' + (data.error || 'Errore sconosciuto'));
            }
        })
        .catch(error => {
            console.error('Errore durante la predizione:', error);
            showError(`Errore predizione: ${error.message}`);
        })
        .finally(() => {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
        });
    });
}

// Inizializzazione training modello
function initModelTraining(trainBtn, problemTypeSelect) {
    if (!trainBtn || !problemTypeSelect) {
        console.error('Elementi per training modello (trainBtn, problemTypeSelect) non trovati.');
        showError('Elementi per training modello non trovati nella pagina');
        return;
    }
    
    // Usa window.problemType per coerenza globale e per assicurare che checkTrainButtonState funzioni correttamente
    if (!window.problemType) {
        window.problemType = problemTypeSelect.value;
    } else {
        // Assicurati che il select rifletta il problemType globale se già settato
        problemTypeSelect.value = window.problemType;
    }
    console.log('Tipo di problema inizializzato a:', window.problemType);
    
    // Rimuovi e riaggiungi l'event listener per problemTypeSelect per evitare duplicazioni
    const newProblemTypeSelect = problemTypeSelect.cloneNode(true);
    problemTypeSelect.parentNode.replaceChild(newProblemTypeSelect, problemTypeSelect);

    newProblemTypeSelect.addEventListener('change', function() {
        window.problemType = newProblemTypeSelect.value;
        console.log('Tipo di problema cambiato a:', window.problemType);
        checkTrainButtonState(); // Verifica lo stato del pulsante TRAIN
    });

    // Rimuovi listener precedenti dal bottone train e aggiungi quello nuovo all'elemento corretto
    const newTrainBtn = trainBtn.cloneNode(true);
    trainBtn.parentNode.replaceChild(newTrainBtn, trainBtn);
    checkTrainButtonState();
    
    // Rimuovi eventuali listener precedenti
    trainBtn.removeEventListener('click', trainModelHandler);
    
    // Aggiungi il nuovo listener
    trainBtn.addEventListener('click', trainModelHandler);
    
    // Funzione di gestione del training
    function trainModelHandler(event) {
        // Previeni comportamenti di default
        event.preventDefault();
        event.stopPropagation();
        
        console.log('Click sul pulsante TRAIN rilevato');
        
        // Verifica che il tipo di problema sia selezionato
        if (!window.problemType) {
            window.problemType = document.getElementById('problem-type').value;
        }
        console.log('Tipo di problema selezionato:', window.problemType);
        
        // Verifica che le colonne del dataset siano state caricate
        console.log('Colonne dataset disponibili:', window.datasetColumns);
        
        // Se non ci sono colonne disponibili, usa le colonne di default
        if (!window.datasetColumns || window.datasetColumns.length === 0) {
            window.datasetColumns = ['colonna1', 'colonna2', 'target'];
            console.log('Usando colonne di default:', window.datasetColumns);
            showToast('Usando colonne di default per il training', 'warning');
        }
        
        // Se non c'è una colonna target selezionata, usa l'ultima colonna disponibile
        if (!window.targetColumn) {
            window.targetColumn = window.datasetColumns[window.datasetColumns.length - 1]; // Usa l'ultima colonna come target di default
            console.log('Colonna target impostata automaticamente:', window.targetColumn);
            showToast(`Colonna target impostata automaticamente a: ${window.targetColumn}`, 'info');
        }
        
        // Mostra loader e disabilita pulsante
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="inline-block h-5 w-5 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></span> Training in corso...';
        
        // Log per debug
        console.log('Invio richiesta di training:');
        console.log('- Tipo di problema:', window.problemType);
        console.log('- Colonna target:', window.targetColumn);
        
        // Prepara i dati da inviare
        const formData = new FormData();
        formData.append('target_column', window.targetColumn);
        formData.append('problem_type', window.problemType);
        
        // Aggiungi il file se è stato caricato
        if (window.uploadedFile) {
            formData.append('file', window.uploadedFile);
            console.log('File aggiunto alla richiesta di training:', window.uploadedFile.name);
        } else {
            console.error('Nessun file disponibile per il training');
            showError('Nessun file disponibile per il training. Ricarica il file.');
            trainBtn.disabled = false;
            trainBtn.innerHTML = 'TRAIN';
            return;
        }
        
        // Mostra il loader animato nella sezione training
        const loader = document.getElementById('training-loader');
        if (loader) loader.classList.remove('hidden');
        
        // Invia richiesta al backend
        fetch('/train_model', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Errore HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Risposta training modello:', data);
            if (data.success) {
                // Mostra la sezione dei risultati
                const visualizationSection = document.getElementById('visualization-section');
                visualizationSection.classList.remove('hidden');
                
                // Aggiorna i grafici e le metriche
                if (data.plot_data) {
                    updatePlots(data.plot_data);
                    
                    // Genera lo scatter plot
                    if (data.plot_data.scatter_data) {
                        generateScatterPlot(data.plot_data.scatter_data, window.targetColumn);
                    }
                }
                
                if (data.metrics) {
                    updateMetrics(data.metrics);
                }
                
                // Crea il form di inferenza con tutti i campi tranne la colonna target
                if (window.datasetColumns && window.datasetColumns.length > 0) {
                    createInferenceForm(window.datasetColumns, window.targetColumn, data.feature_info);
                }
                
                // Scorri fino alla sezione di visualizzazione
                setTimeout(() => {
                    visualizationSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 300);
                
                showSuccess('Modello addestrato con successo');
            } else {
                showError('Errore nel training del modello: ' + (data.error || 'Errore sconosciuto'));
            }
        })
        .catch(error => {
            console.error('Errore durante il training del modello:', error);
            showError(error.message || 'Errore durante il training del modello');
        })
        .finally(() => {
            // Ripristina il pulsante
            trainBtn.disabled = false;
            trainBtn.innerHTML = 'TRAIN';
        });
    }
}
