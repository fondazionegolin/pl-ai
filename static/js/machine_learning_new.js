// Variabili globali
let uploadedFile = null;
let datasetColumns = [];
let targetColumn = null;
let problemType = null;
let trainedModel = null;
let scatterPlot = null;
let fitCurve = null;
let datasetData = null;

// Funzione per formattare la dimensione del file
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
    alert('Errore: ' + message);
}

// Funzione per mostrare messaggi di successo
function showSuccess(message) {
    console.log(message);
    alert('Successo: ' + message);
}

// Inizializzazione al caricamento del DOM
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded - Machine Learning JS v2');
    
    // Elementi DOM
    const fileUpload = document.getElementById('dataset-upload');
    const dropZone = document.getElementById('drop-zone');
    const browseBtn = document.getElementById('browse-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const analysisSection = document.getElementById('analysis-section');
    const targetColumnSelect = document.getElementById('target-column');
    const analyzeTargetBtn = document.getElementById('analyze-target');
    const problemTypeSelect = document.getElementById('problem-type');
    const trainBtn = document.getElementById('train-btn');
    
    // Debug - Verifica che gli elementi DOM siano stati trovati
    console.log('File upload element:', fileUpload);
    console.log('Drop zone element:', dropZone);
    console.log('Browse button element:', browseBtn);
    console.log('Analyze button element:', analyzeBtn);
    
    // Inizializzazione UI
    if (fileInfo) {
        fileInfo.classList.add('hidden');
    }
    
    // Gestione tutorial
    initTutorial();
    
    // Gestione caricamento file
    initFileUpload(fileUpload, dropZone, browseBtn, analyzeBtn, fileInfo, fileName, fileSize);
    
    // Gestione analisi dataset
    initDatasetAnalysis(analyzeBtn, analysisSection, targetColumnSelect);
    
    // Gestione analisi colonna target
    initTargetAnalysis(analyzeTargetBtn, targetColumnSelect);
    
    // Gestione training modello
    initModelTraining(trainBtn, problemTypeSelect);
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
function initFileUpload(fileUpload, dropZone, browseBtn, analyzeBtn, fileInfo, fileName, fileSize) {
    if (!fileUpload || !dropZone || !browseBtn) {
        console.error('Elementi DOM per il caricamento file non trovati');
        return;
    }
    
    console.log('Inizializzazione caricamento file...');
    
    // Gestione input file
    fileUpload.addEventListener('change', function(e) {
        console.log('File input changed');
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0], fileName, fileSize, fileInfo, analyzeBtn);
        }
    });
    
    // Gestione click sul pulsante "sfoglia"
    browseBtn.addEventListener('click', function() {
        console.log('Browse button clicked');
        fileUpload.click();
    });
    
    // Gestione drag and drop
    setupDragAndDrop(dropZone, fileName, fileSize, fileInfo, analyzeBtn);
}

// Gestione selezione file
function handleFileSelection(file, fileName, fileSize, fileInfo, analyzeBtn) {
    console.log('File selezionato:', file.name, file.type);
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showError('Il file deve essere in formato CSV');
        return;
    }
    
    uploadedFile = file;
    
    // Mostra info file
    fileName.textContent = uploadedFile.name;
    fileSize.textContent = formatFileSize(uploadedFile.size);
    fileInfo.classList.remove('hidden');
    
    // Abilita pulsante analisi
    analyzeBtn.disabled = false;
    
    showSuccess('File caricato con successo: ' + file.name);
}

// Setup drag and drop
function setupDragAndDrop(dropZone, fileName, fileSize, fileInfo, analyzeBtn) {
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
            handleFileSelection(dt.files[0], fileName, fileSize, fileInfo, analyzeBtn);
        }
    }, false);
}

// Inizializzazione analisi dataset
function initDatasetAnalysis(analyzeBtn, analysisSection, targetColumnSelect) {
    if (!analyzeBtn) {
        console.error('Pulsante analisi non trovato');
        return;
    }
    
    analyzeBtn.addEventListener('click', function() {
        console.log('Analyze button clicked');
        if (!uploadedFile) {
            showError('Nessun file selezionato');
            return;
        }
        
        // Mostra loader
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="inline-block h-5 w-5 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></span> Analisi in corso...';
        
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        console.log('Invio file al server per analisi:', uploadedFile.name);
        
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
            console.log('Risposta analisi:', data);
            if (data.success) {
                // Popola la descrizione
                document.getElementById('dataset-description').innerHTML = data.description;
                
                // Popola il dropdown delle colonne
                datasetColumns = data.columns;
                targetColumnSelect.innerHTML = '<option value="" disabled selected>Seleziona una colonna</option>';
                
                datasetColumns.forEach(column => {
                    const option = document.createElement('option');
                    option.value = column;
                    option.textContent = column;
                    targetColumnSelect.appendChild(option);
                });
                
                // Mostra la sezione di analisi
                analysisSection.classList.remove('hidden');
                
                // Salva i dati del dataset
                datasetData = data.data;
                
                showSuccess('Dataset analizzato con successo');
            } else {
                showError('Errore nell\'analisi del dataset: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Errore durante l\'analisi del dataset:', error);
            showError(error.message || 'Errore durante l\'analisi del dataset');
        })
        .finally(() => {
            // Ripristina il pulsante
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analizza Dataset';
        });
    });
}

// Inizializzazione analisi colonna target
function initTargetAnalysis(analyzeTargetBtn, targetColumnSelect) {
    if (!analyzeTargetBtn || !targetColumnSelect) {
        console.error('Elementi per analisi target non trovati');
        return;
    }
    
    targetColumnSelect.addEventListener('change', function() {
        analyzeTargetBtn.disabled = false;
    });
    
    analyzeTargetBtn.addEventListener('click', function() {
        targetColumn = targetColumnSelect.value;
        if (!targetColumn) {
            showError('Seleziona una colonna target');
            return;
        }
        
        // Implementa l'analisi della colonna target qui
        console.log('Analisi della colonna target:', targetColumn);
    });
}

// Inizializzazione training modello
function initModelTraining(trainBtn, problemTypeSelect) {
    if (!trainBtn || !problemTypeSelect) {
        console.error('Elementi per training modello non trovati');
        return;
    }
    
    problemTypeSelect.addEventListener('change', function() {
        problemType = problemTypeSelect.value;
        if (problemType && targetColumn) {
            trainBtn.disabled = false;
        }
    });
    
    trainBtn.addEventListener('click', function() {
        if (!problemType || !targetColumn) {
            showError('Seleziona il tipo di problema e la colonna target');
            return;
        }
        
        // Implementa il training del modello qui
        console.log('Training del modello:', problemType, targetColumn);
    });
}
