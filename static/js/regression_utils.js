// Funzioni di utilità per la pagina di regressione

// Funzione per filtrare i dataset in base al modello selezionato
function filterDatasetsByModel() {
    try {
        // Ottieni il tipo di modello selezionato
        const modelTypeElement = document.querySelector('input[name="model-type"]:checked');
        if (!modelTypeElement) {
            console.error('Elemento model-type non trovato');
            return;
        }
        const modelType = modelTypeElement.value;
        const datasetSelect = document.getElementById('dataset-select');
        if (!datasetSelect) {
            console.error('Elemento dataset-select non trovato');
            return;
        }
        
        console.log('Filtrando dataset per modello:', modelType);
        
        // Nascondi tutte le opzioni
        Array.from(datasetSelect.options).forEach(option => {
            option.style.display = 'none';
        });
        
        // Mostra solo le opzioni pertinenti al modello selezionato
        const datasetPrefix = modelType + '-';
        Array.from(datasetSelect.options).forEach(option => {
            if (option.value.startsWith(datasetPrefix) || option.value === '') {
                option.style.display = '';
            }
        });
        
        // Seleziona la prima opzione visibile
        for (let i = 0; i < datasetSelect.options.length; i++) {
            if (datasetSelect.options[i].style.display !== 'none') {
                datasetSelect.selectedIndex = i;
                break;
            }
        }
    } catch (error) {
        console.error('Errore in filterDatasetsByModel:', error);
    }
}

// Alias per compatibilità
window.filterDatasetsByModelLocal = filterDatasetsByModel;

// Funzione di logging per debug
function logToConsole(message) {
    console.log('[DEBUG]', message);
}

// Funzione per mostrare messaggi toast
function showToast(message, type = 'info') {
    console.log(`[TOAST] ${type}: ${message}`);
    // Implementazione del toast UI se necessario
}

// Esporta le funzioni
window.filterDatasetsByModel = filterDatasetsByModel;
window.filterDatasetsByModelLocal = filterDatasetsByModel;
window.logToConsole = logToConsole;
window.showToast = showToast;
