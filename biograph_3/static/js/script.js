// Este código só deve rodar se estivermos na página do grafo
document.addEventListener('DOMContentLoaded', () => {
    const iframe = document.getElementById('graph-iframe');

    // Verifica se o iframe existe na página atual
    if (iframe) {
        // Pega a query da URL (ex: ?query=mars)
        const urlParams = new URLSearchParams(window.location.search);
        const query = urlParams.get('query');

        // Define o src do iframe para a nossa rota no backend, passando a query
        if (query) {
            iframe.src = `/get_graph_data?query=${encodeURIComponent(query)}`;
        } else {
            iframe.src = '/get_graph_data';
        }
    }
});