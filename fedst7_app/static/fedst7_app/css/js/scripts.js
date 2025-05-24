document.addEventListener('DOMContentLoaded', function() {
    // Dynamic form handling
    const predictionForms = document.querySelectorAll('.prediction-form');
    
    predictionForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const submitBtn = this.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            
            // Simulate processing delay
            setTimeout(() => {
                this.submit();
            }, 1000);
        });
    });
    
    // Model type badges
    const modelTypeBadges = document.querySelectorAll('.model-type-badge');
    
    modelTypeBadges.forEach(badge => {
        const type = badge.textContent.trim().toLowerCase();
        let bgClass = 'bg-secondary';
        
        if (type.includes('regression')) bgClass = 'bg-primary';
        if (type.includes('classification')) bgClass = 'bg-success';
        if (type.includes('clustering')) bgClass = 'bg-info';
        
        badge.classList.add(bgClass);
    });
});