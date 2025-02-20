// Verifica se l'utente è sulla landing page
function isLandingPage() {
    return window.location.pathname === '/';
}

// Intercetta tutte le richieste e aggiungi il token di autenticazione
function addAuthorizationHeader(url, options = {}) {
    const token = sessionStorage.getItem('authToken');
    if (token) {
        if (!options.headers) {
            options.headers = {};
        }
        options.headers['Authorization'] = `Bearer ${token}`;
    }
    return { url, options };
}

// Funzione per verificare se l'utente è autenticato
async function checkAuth() {
    // Non verificare l'autenticazione sulla landing page
    if (isLandingPage()) {
        return true;
    }

    const token = sessionStorage.getItem('authToken');
    if (!token) {
        window.location.replace('/');
        return false;
    }

    try {
        const response = await fetch('/verify-token', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            sessionStorage.removeItem('authToken');
            window.location.replace('/');
            return false;
        }

        return true;
    } catch (error) {
        console.error('Auth check error:', error);
        sessionStorage.removeItem('authToken');
        window.location.replace('/');
        return false;
    }
}

// Intercetta tutte le richieste fetch
const originalFetch = window.fetch;
window.fetch = async function(url, options = {}) {
    // Non intercettare le richieste sulla landing page
    if (isLandingPage()) {
        return originalFetch(url, options);
    }

    const { url: modifiedUrl, options: modifiedOptions } = addAuthorizationHeader(url, options);
    const response = await originalFetch(modifiedUrl, modifiedOptions);
    
    // Se riceviamo un 401, significa che il token non è valido
    if (response.status === 401) {
        sessionStorage.removeItem('authToken');
        window.location.replace('/');
        return new Response(null, { status: 401 });
    }
    
    return response;
};

// Funzione di logout
async function logout() {
    try {
        sessionStorage.removeItem('authToken');
        window.location.replace('/');
    } catch (error) {
        console.error('Logout error:', error);
    }
}

// Funzione per navigare a una pagina protetta
async function navigateToProtectedPage(url) {
    const token = sessionStorage.getItem('authToken');
    if (!token) {
        window.location.replace('/');
        return;
    }

    try {
        // Prima verifica il token
        const verifyResponse = await fetch('/verify-token', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!verifyResponse.ok) {
            sessionStorage.removeItem('authToken');
            window.location.replace('/');
            return;
        }

        // Se il token è valido, procedi con la navigazione
        window.location.href = url;
    } catch (error) {
        console.error('Navigation error:', error);
        window.location.replace('/');
    }
}

// Funzione per aggiornare il profilo
async function updateProfile(formData) {
    try {
        const response = await fetch('/api/profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                firstName: formData.get('firstName'),
                lastName: formData.get('lastName'),
                bio: formData.get('bio')
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            // Mostra un messaggio di successo
            showMessage('success', 'Profilo aggiornato con successo!');
            // Aggiorna i campi del profilo nella pagina
            updateProfileFields(data.data);
        } else {
            showMessage('error', data.message || 'Errore durante laggiornamento del profilo');
        }
        
        return data;
    } catch (error) {
        console.error('Error updating profile:', error);
        showMessage('error', 'Errore di rete durante laggiornamento del profilo');
        throw error;
    }
}

// Funzione per mostrare messaggi all'utente
function showMessage(type, message) {
    const messageDiv = document.getElementById('message');
    const messageText = document.getElementById('messageText');
    if (!messageDiv || !messageText) {
        console.warn('Message elements not found');
        return;
    }
    
    // Imposta il testo del messaggio
    messageText.textContent = message;
    
    // Aggiorna lo stile in base al tipo e l'icona
    const gradient = type === 'success' ? 'from-blue-600 to-indigo-600' : 'from-red-600 to-pink-600';
    const iconPath = type === 'success' 
        ? 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'
        : 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z';
    
    messageDiv.querySelector('.flex').className = `flex items-center p-4 rounded-xl shadow-lg bg-gradient-to-r ${gradient} text-white`;
    messageDiv.querySelector('path').setAttribute('d', iconPath);
    
    // Mostra il messaggio con animazione
    messageDiv.classList.remove('translate-x-full', 'opacity-0');
    messageDiv.classList.add('translate-x-0', 'opacity-100');
    
    // Nascondi il messaggio dopo 3 secondi
    setTimeout(() => {
        messageDiv.classList.remove('translate-x-0', 'opacity-100');
        messageDiv.classList.add('translate-x-full', 'opacity-0');
    }, 3000);
}

// Funzione per aggiornare i campi del profilo nella pagina
function updateProfileFields(profileData) {
    const fields = ['first_name', 'last_name', 'email', 'bio'];
    fields.forEach(field => {
        const element = document.getElementById(field);
        if (element) {
            element.value = profileData[field] || '';
        }
    });
    
    // Aggiorna limmagine del profilo se presente
    const profilePicture = document.getElementById('profile_picture');
    if (profilePicture && profileData.profile_picture) {
        profilePicture.src = profileData.profile_picture;
    }
}

// Funzione per inizializzare gli event listener
function initializeEventListeners() {
    // Intercetta i click sui link protetti
    document.querySelectorAll('a[data-protected="true"]').forEach(link => {
        if (!link.hasAttribute('data-handler-attached')) {
            link.setAttribute('data-handler-attached', 'true');
            link.addEventListener('click', (e) => {
                e.preventDefault();
                navigateToProtectedPage(link.href);
            });
        }
    });

    // Aggiungi event listener per il form del profilo
    const profileForm = document.getElementById('profileForm');
    if (profileForm && !profileForm.hasAttribute('data-handler-attached')) {
        profileForm.setAttribute('data-handler-attached', 'true');
        profileForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(profileForm);
            await updateProfile(formData);
        });
    }

    // Esegui il controllo dell'autenticazione
    checkAuth();
}

// Inizializza gli event listener al caricamento della pagina
document.addEventListener('DOMContentLoaded', initializeEventListeners);
