class ArticlesBrowser {
    constructor() {
        this.articles = [];
        this.filteredArticles = [];
        this.selectedArticles = new Set();
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadArticles();
        
        // Check for URL parameters
        this.handleUrlParameters();
    }

    initializeElements() {
        this.searchInput = document.getElementById('searchInput');
        this.searchBtn = document.getElementById('searchBtn');
        this.clearSearchBtn = document.getElementById('clearSearchBtn');
        this.selectAllBtn = document.getElementById('selectAllBtn');
        this.deselectAllBtn = document.getElementById('deselectAllBtn');
        this.loadSelectedBtn = document.getElementById('loadSelectedBtn');
        this.articlesGrid = document.getElementById('articlesGrid');
        this.resultsTitle = document.getElementById('resultsTitle');
        this.resultsCount = document.getElementById('resultsCount');
        this.selectedCount = document.getElementById('selectedCount');
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    attachEventListeners() {
        // Search functionality
        this.searchBtn.addEventListener('click', this.performSearch.bind(this));
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        this.clearSearchBtn.addEventListener('click', this.clearSearch.bind(this));

        // Selection functionality
        this.selectAllBtn.addEventListener('click', this.selectAll.bind(this));
        this.deselectAllBtn.addEventListener('click', this.deselectAll.bind(this));

        // Load selected articles
        this.loadSelectedBtn.addEventListener('click', this.loadSelectedArticles.bind(this));
    }

    handleUrlParameters() {
        const urlParams = new URLSearchParams(window.location.search);
        
        // Check for search parameter
        const searchQuery = urlParams.get('search');
        if (searchQuery) {
            this.searchInput.value = searchQuery;
            // Wait for articles to load, then search
            setTimeout(() => {
                this.performSearch();
            }, 1000);
        }

        // Check for preselected articles
        const preselected = urlParams.get('preselect');
        if (preselected) {
            const articleIds = preselected.split(',');
            setTimeout(() => {
                this.preselectArticles(articleIds);
            }, 1000);
        }
    }

    async loadArticles() {
        try {
            console.log('Loading articles...');
            const response = await fetch('/ask-lumi/api/articles-list');
            console.log('Response status:', response.status);
            const result = await response.json();
            console.log('Articles result:', result);
            console.log('Success field:', result.success);
            console.log('Articles field:', !!result.articles);

            if (result.success) {
                this.articles = result.articles;
                this.filteredArticles = [...this.articles];
                console.log('Loaded articles count:', this.articles.length);
                console.log('About to render articles...');
                this.renderArticles();
                console.log('About to update results info...');
                this.updateResultsInfo();
                console.log('Articles loading completed successfully');
            } else {
                console.error('API returned success=false:', result);
                this.showError('Failed to load articles');
            }
        } catch (error) {
            console.error('Error loading articles:', error);
            this.showError('Error loading articles: ' + error.message);
        }
    }

    performSearch() {
        const query = this.searchInput.value.trim().toLowerCase();
        
        if (!query) {
            this.clearSearch();
            return;
        }

        const keywords = query.split(/\s+/);
        
        this.filteredArticles = this.articles.filter(article => {
            const searchText = `${article.title || ''} ${article.display_name || ''} ${article.filename || ''}`.toLowerCase();
            return keywords.some(keyword => searchText.includes(keyword));
        });

        this.resultsTitle.textContent = `Search Results for "${query}"`;
        this.renderArticles();
        this.updateResultsInfo();
    }

    clearSearch() {
        this.searchInput.value = '';
        this.filteredArticles = [...this.articles];
        this.resultsTitle.textContent = 'Available Articles';
        this.renderArticles();
        this.updateResultsInfo();
    }

    renderArticles() {
        console.log('renderArticles called, filteredArticles length:', this.filteredArticles.length);
        if (this.filteredArticles.length === 0) {
            console.log('No articles to render, showing empty state');
            this.articlesGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search"></i>
                    <h3>No articles found</h3>
                    <p>Try adjusting your search terms or clear the search to see all articles.</p>
                </div>
            `;
            return;
        }

        console.log('Rendering', this.filteredArticles.length, 'articles...');
        
        // Limit initial render to prevent browser hang
        const articlesToRender = this.filteredArticles.slice(0, 50);
        console.log('Actually rendering first', articlesToRender.length, 'articles');
        
        this.articlesGrid.innerHTML = articlesToRender.map(article => `
            <div class="article-card ${this.selectedArticles.has(article.filename) ? 'selected' : ''}" 
                 data-filename="${article.filename}">
                <input type="checkbox" class="article-checkbox" 
                       ${this.selectedArticles.has(article.filename) ? 'checked' : ''}>
                
                <h4 class="article-title">${article.title || article.display_name}</h4>
                
                <div class="article-links">
                    ${article.download_url ? `
                        <a href="${article.download_url}" target="_blank" class="link-badge pdf">
                            <i class="fas fa-file-pdf"></i> PDF
                        </a>
                    ` : ''}
                    
                    ${article.pmc_link ? `
                        <a href="${article.pmc_link}" target="_blank" class="link-badge pmc">
                            <i class="fas fa-external-link-alt"></i> PMC
                        </a>
                    ` : ''}
                </div>
            </div>
        `).join('');
        
        // Add pagination info if there are more articles
        if (this.filteredArticles.length > 50) {
            this.articlesGrid.innerHTML += `
                <div class="pagination-info">
                    <p>Showing first 50 of ${this.filteredArticles.length} articles. Use search to narrow results.</p>
                </div>
            `;
        }

        // Add click handlers to cards
        document.querySelectorAll('.article-card').forEach(card => {
            card.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox' && !e.target.closest('a')) {
                    const checkbox = card.querySelector('.article-checkbox');
                    checkbox.checked = !checkbox.checked;
                    this.toggleArticle(card.dataset.filename);
                }
            });
            
            // Add direct checkbox handler
            const checkbox = card.querySelector('.article-checkbox');
            checkbox.addEventListener('change', (e) => {
                this.toggleArticle(card.dataset.filename);
            });
        });
    }

    toggleArticle(filename) {
        if (this.selectedArticles.has(filename)) {
            this.selectedArticles.delete(filename);
        } else {
            this.selectedArticles.add(filename);
        }

        this.updateSelectionUI();
        this.updateLoadButton();
    }

    selectAll() {
        this.filteredArticles.forEach(article => {
            this.selectedArticles.add(article.filename);
        });
        this.updateSelectionUI();
        this.updateLoadButton();
    }

    deselectAll() {
        this.selectedArticles.clear();
        this.updateSelectionUI();
        this.updateLoadButton();
    }

    preselectArticles(articleIds) {
        articleIds.forEach(id => {
            const article = this.articles.find(a => a.filename === id || a.display_name === id);
            if (article) {
                this.selectedArticles.add(article.filename);
            }
        });
        this.updateSelectionUI();
        this.updateLoadButton();
    }

    updateSelectionUI() {
        // Update card selection states
        document.querySelectorAll('.article-card').forEach(card => {
            const filename = card.dataset.filename;
            const checkbox = card.querySelector('.article-checkbox');
            const isSelected = this.selectedArticles.has(filename);
            
            card.classList.toggle('selected', isSelected);
            checkbox.checked = isSelected;
        });
    }

    updateLoadButton() {
        const count = this.selectedArticles.size;
        this.selectedCount.textContent = count;
        this.loadSelectedBtn.disabled = count === 0;
    }

    updateResultsInfo() {
        const total = this.filteredArticles.length;
        const selected = this.selectedArticles.size;
        
        if (this.searchInput.value.trim()) {
            this.resultsCount.textContent = `${total} articles found`;
        } else {
            this.resultsCount.textContent = `${total} articles available`;
        }
    }

    async loadSelectedArticles() {
        if (this.selectedArticles.size === 0) return;

        this.showLoading();

        try {
            const filenames = Array.from(this.selectedArticles);
            const response = await fetch('/ask-lumi/api/load-articles', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filenames })
            });

            const result = await response.json();

            if (result.success) {
                // Redirect to main page with success message
                const params = new URLSearchParams({
                    loaded: result.loaded_count,
                    message: `Successfully loaded ${result.loaded_count} articles`,
                    session_id: result.session_id
                });
                window.location.href = `/ask-lumi?${params.toString()}`;
            } else {
                alert('Error loading articles: ' + result.error);
            }
        } catch (error) {
            console.error('Error loading articles:', error);
            alert('Error loading articles: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    showError(message) {
        this.articlesGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }

    showLoading() {
        this.loadingOverlay.classList.remove('hidden');
    }

    hideLoading() {
        this.loadingOverlay.classList.add('hidden');
    }
}

// Global reference for event handlers
let articlesBrowser;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    articlesBrowser = new ArticlesBrowser();
});