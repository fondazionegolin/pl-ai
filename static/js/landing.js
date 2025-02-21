// Funzione per gestire il logout e reindirizzare alla landing page
async function handleLogout() {
    try {
        // Rimuovi il token di autenticazione
        sessionStorage.removeItem('authToken');
        
        // Effettua una richiesta al backend per il logout
        await fetch('/auth/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // Reindirizza alla landing page
        window.location.href = '/';
    } catch (error) {
        console.error('Logout error:', error);
        // In caso di errore, reindirizza comunque alla landing page
        window.location.href = '/';
    }
}

// Esponi la funzione globalmente
window.handleLogout = handleLogout;
