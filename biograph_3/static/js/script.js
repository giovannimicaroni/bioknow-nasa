// Global variable to store the preloaded blob URL
let preloadedBlobUrl = null;

// Preload graph data on landing page
if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
    console.log('Starting graph preload...');
    
    fetch('/get_graph_data')
        .then(response => response.blob())
        .then(blob => {
            // Create a blob URL that we can reuse
            preloadedBlobUrl = URL.createObjectURL(blob);
            // Store the blob URL in sessionStorage
            sessionStorage.setItem('graphBlobUrl', preloadedBlobUrl);
            console.log('Graph preloaded successfully! Blob URL:', preloadedBlobUrl);
        })
        .catch(error => {
            console.error('Preload failed:', error);
        });
}

// Load graph on graph page
document.addEventListener('DOMContentLoaded', () => {
    const iframe = document.getElementById('graph-iframe');

    if (iframe) {
        const urlParams = new URLSearchParams(window.location.search);
        const query = urlParams.get('query');

        // If there's a search query, load that specific graph
        if (query && query !== 'N/A' && query !== '') {
            console.log('Loading searched graph for:', query);
            iframe.src = `/get_graph_data?query=${encodeURIComponent(query)}`;
        } 
        // Try to use preloaded blob URL
        else {
            const cachedBlobUrl = sessionStorage.getItem('graphBlobUrl');
            
            if (cachedBlobUrl) {
                console.log('Using preloaded graph from blob URL!');
                iframe.src = cachedBlobUrl;
            } else {
                console.log('No cache found, loading normally...');
                iframe.src = '/get_graph_data';
            }
        }

        // Optional: Add loading indicator
        iframe.addEventListener('load', () => {
            console.log('Graph loaded successfully!');
        });
    }
});