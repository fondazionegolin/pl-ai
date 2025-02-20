// Profile Menu
function toggleProfileMenu() {
    const dropdown = document.getElementById('profileDropdown');
    dropdown.classList.toggle('hidden');
}

// Chiudi il menu quando si clicca fuori
document.addEventListener('click', function(event) {
    const profileMenu = document.getElementById('profileMenu');
    const dropdown = document.getElementById('profileDropdown');
    
    if (profileMenu && !profileMenu.contains(event.target)) {
        dropdown.classList.add('hidden');
    }
});
