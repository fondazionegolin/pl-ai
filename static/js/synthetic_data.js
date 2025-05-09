/**
 * Funzioni per la generazione di dataset sintetici per diversi algoritmi di regressione
 */

// Funzione per generare dataset sintetici
window.generateSyntheticData = function(datasetType, size = 50) {
    const data = [];
    const header = [];
    
    // Imposta i nomi delle colonne in base al tipo di dataset
    switch(datasetType) {
        case 'linear-1':
            header.push('Metri Quadri', 'Prezzo (€)');
            // Genera dati lineari per metri quadri vs prezzo
            for (let i = 0; i < size; i++) {
                const x = Math.round(30 + Math.random() * 170); // Metri quadri tra 30 e 200
                const y = Math.round((x * 1500) + (Math.random() * 20000 - 10000)); // Prezzo con un po' di rumore
                data.push([x, y]);
            }
            break;
            
        case 'linear-2':
            header.push('Budget Marketing (€)', 'Vendite (€)');
            // Genera dati lineari per budget marketing vs vendite
            for (let i = 0; i < size; i++) {
                const x = Math.round(1000 + Math.random() * 9000); // Budget tra 1000 e 10000
                const y = Math.round((x * 2.5) + (Math.random() * 5000 - 2500)); // Vendite con rumore
                data.push([x, y]);
            }
            break;
            
        case 'linear-3':
            header.push('Ore di Studio', 'Voto Esame');
            // Genera dati lineari per ore di studio vs voto esame
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 9); // Ore tra 1 e 10
                const y = Math.min(30, Math.max(18, Math.round((x * 1.5) + 15 + (Math.random() * 4 - 2)))); // Voto con rumore
                data.push([x, y]);
            }
            break;
            
        case 'polynomial-1':
            header.push('Temperatura (°C)', 'Efficienza (%)');
            // Genera dati polinomiali per temperatura vs efficienza
            for (let i = 0; i < size; i++) {
                const x = Math.round(-10 + Math.random() * 50); // Temperatura tra -10 e 40
                // Funzione parabolica con massimo intorno a 20°C
                const y = Math.round(90 - 0.1 * Math.pow(x - 20, 2) + (Math.random() * 6 - 3));
                data.push([x, Math.max(0, Math.min(100, y))]);
            }
            break;
            
        case 'polynomial-2':
            header.push('Velocità (km/h)', 'Consumo (l/100km)');
            // Genera dati polinomiali per velocità vs consumo carburante
            for (let i = 0; i < size; i++) {
                const x = Math.round(20 + Math.random() * 130); // Velocità tra 20 e 150 km/h
                // Curva a U per il consumo (alto a basse e alte velocità)
                const y = 5 + 0.0008 * Math.pow(x - 80, 2) + (Math.random() * 2 - 1);
                // Arrotonda a 1 decimale e converte in numero
                const roundedY = Math.round(y * 10) / 10;
                data.push([x, roundedY]);
            }
            break;
            
        case 'polynomial-3':
            header.push('Dose (mg)', 'Risposta (%)');
            // Genera dati polinomiali per dose vs risposta farmacologica
            for (let i = 0; i < size; i++) {
                const x = Math.round(5 + Math.random() * 95); // Dose tra 5 e 100 mg
                // Curva a campana per la risposta
                const y = Math.round(90 * Math.exp(-0.0005 * Math.pow(x - 50, 2)) + (Math.random() * 10 - 5));
                data.push([x, Math.max(0, Math.min(100, y))]);
            }
            break;
            
        case 'svr-1':
            header.push('Variabile X', 'Variabile Y');
            // Genera dati con outlier per SVR
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99); // X tra 1 e 100
                let y = 2 * x + 10 + (Math.random() * 10 - 5); // Base lineare con rumore
                
                // Aggiungi alcuni outlier
                if (Math.random() < 0.1) {
                    y += (Math.random() > 0.5 ? 1 : -1) * Math.random() * 50;
                }
                
                data.push([x, Math.round(y)]);
            }
            break;
            
        case 'svr-2':
            header.push('Tempo', 'Valore');
            // Genera dati sinusoidali per SVR
            for (let i = 0; i < size; i++) {
                const x = i * (10 / size); // X tra 0 e 10
                const y = 20 * Math.sin(x) + (Math.random() * 6 - 3); // Funzione seno con rumore
                // Arrotonda a 2 decimali e converte in numero
                const roundedX = Math.round(x * 100) / 100;
                const roundedY = Math.round(y * 100) / 100;
                data.push([roundedX, roundedY]);
            }
            break;
            
        case 'svr-3':
            header.push('Input', 'Output');
            // Genera dati con rumore per SVR
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99); // X tra 1 e 100
                const y = 50 + 0.5 * x + 10 * Math.sin(x / 5) + (Math.random() * 20 - 10); // Funzione con rumore
                // Arrotonda a 1 decimale e converte in numero
                const roundedY = Math.round(y * 10) / 10;
                data.push([x, roundedY]);
            }
            break;
            
        case 'random_forest-1':
            header.push('Variabile X', 'Variabile Y');
            // Genera dati complessi per Random Forest
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99); // X tra 1 e 100
                let y;
                
                // Funzione a tratti con diverse relazioni
                if (x < 30) {
                    y = 2 * x + (Math.random() * 10 - 5);
                } else if (x < 60) {
                    y = 100 - x + (Math.random() * 15 - 7.5);
                } else {
                    y = 0.05 * x * x - 3 * x + 200 + (Math.random() * 20 - 10);
                }
                
                data.push([x, Math.round(y)]);
            }
            break;
            
        case 'random_forest-2':
            header.push('Variabile X', 'Variabile Y');
            // Genera dati complessi per Random Forest
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99); // X tra 1 e 100
                
                // Funzione a gradini con rumore
                let y;
                if (x < 20) y = 20 + (Math.random() * 10 - 5);
                else if (x < 40) y = 40 + (Math.random() * 10 - 5);
                else if (x < 60) y = 60 + (Math.random() * 10 - 5);
                else if (x < 80) y = 80 + (Math.random() * 10 - 5);
                else y = 100 + (Math.random() * 10 - 5);
                
                data.push([x, Math.round(y)]);
            }
            break;
            
        case 'decision_tree-1':
            header.push('Variabile X', 'Variabile Y');
            // Genera dati a gradini per Decision Tree
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99); // X tra 1 e 100
                
                // Funzione a gradini netti
                let y;
                if (x < 20) y = 10 + (Math.random() * 4 - 2);
                else if (x < 40) y = 30 + (Math.random() * 4 - 2);
                else if (x < 60) y = 20 + (Math.random() * 4 - 2);
                else if (x < 80) y = 50 + (Math.random() * 4 - 2);
                else y = 40 + (Math.random() * 4 - 2);
                
                data.push([x, Math.round(y)]);
            }
            break;
            
        case 'decision_tree-2':
            header.push('Variabile X', 'Variabile Y');
            // Genera dati discontinui per Decision Tree
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99); // X tra 1 e 100
                
                // Funzione con discontinuità
                let y;
                if (x < 25) y = 0.5 * x + (Math.random() * 5 - 2.5);
                else if (x < 50) y = -0.5 * x + 50 + (Math.random() * 5 - 2.5);
                else if (x < 75) y = 0.8 * x - 15 + (Math.random() * 5 - 2.5);
                else y = -0.3 * x + 100 + (Math.random() * 5 - 2.5);
                
                data.push([x, Math.round(y)]);
            }
            break;
            
        default:
            // Dataset predefinito
            header.push('X', 'Y');
            for (let i = 0; i < size; i++) {
                const x = Math.round(1 + Math.random() * 99);
                const y = 2 * x + (Math.random() * 10 - 5);
                data.push([x, Math.round(y)]);
            }
    }
    
    // Converti in formato CSV con gestione speciale dei numeri
    let csvContent = header.join(',') + '\n';
    data.forEach(row => {
        // Assicurati che i numeri siano formattati correttamente
        const formattedRow = row.map(value => {
            if (typeof value === 'number') {
                // Converti i numeri in stringhe con punto decimale
                return String(value).replace(',', '.');
            }
            return value;
        });
        csvContent += formattedRow.join(',') + '\n';
    });
    
    return csvContent;
}

// Filtra i dataset in base al modello selezionato
window.filterDatasetsByModel = function() {
    const selectedModel = document.querySelector('input[name="model-type"]:checked').value;
    const datasetSelect = document.getElementById('example-dataset');
    const options = datasetSelect.querySelectorAll('option');
    
    // Resetta la selezione
    datasetSelect.selectedIndex = 0;
    
    // Mostra/nascondi opzioni in base al modello selezionato
    options.forEach(option => {
        if (option.value === '' || option.getAttribute('data-model') === selectedModel) {
            option.style.display = '';
        } else {
            option.style.display = 'none';
        }
    });
}
