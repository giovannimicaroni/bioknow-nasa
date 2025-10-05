// Global variables
let preloadedBlobUrl = null;
let allKeywords = [];
let selectedKeywords = new Set();
let customKeywords = new Set(); // ADD THIS LINE

// Preload graph data on landing page
if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
    console.log('Starting graph preload...');
    
    fetch('/get_graph_data')
        .then(response => response.blob())
        .then(blob => {
            preloadedBlobUrl = URL.createObjectURL(blob);
            sessionStorage.setItem('graphBlobUrl', preloadedBlobUrl);
            console.log('Graph preloaded successfully!');
        })
        .catch(error => {
            console.error('Preload failed:', error);
        });
}

// Load graph on graph page
document.addEventListener('DOMContentLoaded', () => {
    const iframe = document.getElementById('graph-iframe');

    if (iframe) {
        initializeFilters();
        loadInitialGraph();
    }
});

function loadInitialGraph() {
    const iframe = document.getElementById('graph-iframe');
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('query');

    if (query && query !== 'N/A' && query !== '') {
        console.log('Loading searched graph for:', query);
        iframe.src = `/get_graph_data?query=${encodeURIComponent(query)}`;
    } else {
        const cachedBlobUrl = sessionStorage.getItem('graphBlobUrl');
        
        if (cachedBlobUrl) {
            console.log('Using preloaded graph from blob URL!');
            iframe.src = cachedBlobUrl;
        } else {
            console.log('No cache found, loading normally...');
            iframe.src = '/get_graph_data';
        }
    }

    iframe.addEventListener('load', () => {
        console.log('Graph loaded successfully!');
    });
}

function initializeFilters() {
    // Load keywords from API
    loadKeywords();
    
    // Setup event listeners
    setupFilterListeners();
    
    // Setup custom keywords listeners
    setupCustomKeywordsListeners();
}

function setupCustomKeywordsListeners() {
    const customInput = document.getElementById('custom-keyword-input');
    const addBtn = document.getElementById('add-custom-keyword');
    
    if (customInput && addBtn) {
        // Add keyword on button click
        addBtn.addEventListener('click', () => {
            addCustomKeyword();
        });
        
        // Add keyword on Enter key
        customInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addCustomKeyword();
            }
        });
    }
}

function addCustomKeyword() {
    const input = document.getElementById('custom-keyword-input');
    const keyword = input.value.trim();
    
    if (keyword && keyword.length > 0) {
        customKeywords.add(keyword);
        input.value = '';
        renderCustomKeywords();
        updateSelectedCount();
    }
}

function removeCustomKeyword(keyword) {
    customKeywords.delete(keyword);
    renderCustomKeywords();
    updateSelectedCount();
}

function renderCustomKeywords() {
    const container = document.getElementById('custom-keywords-container');
    
    if (!container) return;
    
    if (customKeywords.size === 0) {
        container.innerHTML = '';
        return;
    }
    
    container.innerHTML = '';
    
    customKeywords.forEach(keyword => {
        const tag = document.createElement('div');
        tag.className = 'custom-keyword-tag';
        tag.innerHTML = `
            <span>${keyword}</span>
            <span class="remove-custom" onclick="removeCustomKeyword('${keyword}')">&times;</span>
        `;
        container.appendChild(tag);
    });
}

function loadKeywords() {
    const keywordsList = document.getElementById('keywords-list');
    
    fetch('/api/keywords')
        .then(response => response.json())
        .then(data => {
            // Check if data is categorized (object) or flat list (array)
            if (Array.isArray(data)) {
                // Flat list - render normally
                allKeywords = data;
                renderKeywords(data);
            } else {
                // Categorized - render with categories
                renderCategorizedKeywords(data);
            }
        })
        .catch(error => {
            console.error('Failed to load keywords:', error);
            keywordsList.innerHTML = '<div class="loading">Failed to load keywords</div>';
        });
}

function renderKeywords(keywords) {
    const keywordsList = document.getElementById('keywords-list');
    
    if (keywords.length === 0) {
        keywordsList.innerHTML = '<div class="loading">No keywords available</div>';
        return;
    }
    
    keywordsList.innerHTML = '';
    
    keywords.forEach(keyword => {
        const item = createKeywordItem(keyword);
        keywordsList.appendChild(item);
    });
}

function renderCategorizedKeywords(categorizedData) {
    const keywordsList = document.getElementById('keywords-list');
    keywordsList.innerHTML = '';
    
    // Flatten all keywords for search functionality
    allKeywords = [];
    Object.values(categorizedData).forEach(keywords => {
        allKeywords.push(...keywords);
    });
    
    // Render each category
    Object.entries(categorizedData).forEach(([category, keywords]) => {
        // Create category header
        const categoryHeader = document.createElement('div');
        categoryHeader.className = 'category-header';
        
        const toggleIcon = document.createElement('span');
        toggleIcon.className = 'category-toggle';
        toggleIcon.textContent = '▼';
        
        const categoryTitle = document.createElement('span');
        categoryTitle.className = 'category-title';
        categoryTitle.textContent = category;
        
        categoryHeader.appendChild(toggleIcon);
        categoryHeader.appendChild(categoryTitle);
        
        // Create category content container
        const categoryContent = document.createElement('div');
        categoryContent.className = 'category-content';
        categoryContent.dataset.category = category;
        
        // Add keywords to category
        keywords.forEach(keyword => {
            const item = createKeywordItem(keyword, category);
            categoryContent.appendChild(item);
        });
        
        // Toggle category collapse/expand
        categoryHeader.addEventListener('click', () => {
            categoryContent.classList.toggle('collapsed');
            toggleIcon.textContent = categoryContent.classList.contains('collapsed') ? '▶' : '▼';
        });
        
        keywordsList.appendChild(categoryHeader);
        keywordsList.appendChild(categoryContent);
    });
}

function createKeywordItem(keyword, category = null) {
    const item = document.createElement('div');
    item.className = 'keyword-item';
    item.dataset.keyword = keyword;
    if (category) {
        item.dataset.category = category;
    }
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'keyword-checkbox';
    checkbox.id = `kw-${keyword}`;
    checkbox.checked = selectedKeywords.has(keyword);
    
    const label = document.createElement('label');
    label.className = 'keyword-text';
    label.htmlFor = `kw-${keyword}`;
    label.textContent = keyword;
    
    item.appendChild(checkbox);
    item.appendChild(label);
    
    // Click on item toggles checkbox
    item.addEventListener('click', (e) => {
        if (e.target !== checkbox) {
            checkbox.checked = !checkbox.checked;
        }
        toggleKeywordSelection(keyword, checkbox.checked);
    });
    
    checkbox.addEventListener('change', (e) => {
        toggleKeywordSelection(keyword, e.target.checked);
    });
    
    return item;
}

function toggleKeywordSelection(keyword, isSelected) {
    if (isSelected) {
        selectedKeywords.add(keyword);
    } else {
        selectedKeywords.delete(keyword);
    }
    updateSelectedCount();
}

function updateSelectedCount() {
    const countBadge = document.getElementById('selected-count');
    const totalSelected = selectedKeywords.size + customKeywords.size;
    countBadge.textContent = `${totalSelected} selected`;
}

function setupFilterListeners() {
    // Toggle filter panel - CLOSE button
    const toggleBtn = document.getElementById('toggle-filters');
    const filterPanel = document.getElementById('filter-panel');
    const openBtn = document.getElementById('open-filters');
    
    // Close filter panel
    toggleBtn.addEventListener('click', () => {
        filterPanel.classList.add('hidden');
        openBtn.classList.remove('hidden');
    });
    
    // Open filter panel
    openBtn.addEventListener('click', () => {
        filterPanel.classList.remove('hidden');
        openBtn.classList.add('hidden');
    });
    
    // Start with panel open
    filterPanel.classList.remove('hidden');
    openBtn.classList.add('hidden');
    
    // Keyword search
    const searchInput = document.getElementById('keyword-search');
    searchInput.addEventListener('input', (e) => {
        filterKeywordsList(e.target.value);
    });
    
    // Apply filters button
    const applyBtn = document.getElementById('apply-filters');
    applyBtn.addEventListener('click', applyFilters);
    
    // Clear filters button
    const clearBtn = document.getElementById('clear-filters');
    clearBtn.addEventListener('click', clearFilters);
}

function filterKeywordsList(searchTerm) {
    const items = document.querySelectorAll('.keyword-item');
    const categories = document.querySelectorAll('.category-content');
    const term = searchTerm.toLowerCase();
    
    // Filter individual items
    items.forEach(item => {
        const keyword = item.dataset.keyword.toLowerCase();
        if (keyword.includes(term)) {
            item.classList.remove('hidden');
        } else {
            item.classList.add('hidden');
        }
    });
    
    // Hide empty categories
    categories.forEach(category => {
        const visibleItems = category.querySelectorAll('.keyword-item:not(.hidden)');
        const categoryHeader = category.previousElementSibling;
        
        if (visibleItems.length === 0) {
            category.classList.add('hidden');
            if (categoryHeader && categoryHeader.classList.contains('category-header')) {
                categoryHeader.classList.add('hidden');
            }
        } else {
            category.classList.remove('hidden');
            if (categoryHeader && categoryHeader.classList.contains('category-header')) {
                categoryHeader.classList.remove('hidden');
            }
        }
    });
}

function applyFilters() {
    const iframe = document.getElementById('graph-iframe');
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('query') || '';
    
    // Build URL with filters
    let url = '/get_graph_data';
    const params = new URLSearchParams();
    
    // Add search query if exists
    if (query && query !== 'N/A') {
        params.append('query', query);
    }
    
    // Combine selected keywords from checkboxes AND custom keywords
    const allSelectedKeywords = new Set();
    
    // Add checked keywords from the list
    selectedKeywords.forEach(keyword => {
        allSelectedKeywords.add(keyword);
    });
    
    // Add custom keywords
    customKeywords.forEach(keyword => {
        allSelectedKeywords.add(keyword);
    });
    
    // Add all keywords as filter parameters
    allSelectedKeywords.forEach(keyword => {
        params.append('filter_keywords[]', keyword.trim());
    });
    
    if (params.toString()) {
        url += '?' + params.toString();
    }
    
    console.log('Applying filters with URL:', url);
    console.log('Selected keywords:', Array.from(selectedKeywords));
    console.log('Custom keywords:', Array.from(customKeywords));
    console.log('All keywords being sent:', Array.from(allSelectedKeywords));
    
    // Reload iframe with new filters
    iframe.src = url;
}

function clearFilters() {
    // Uncheck all checkboxes
    selectedKeywords.clear();
    customKeywords.clear();
    
    const checkboxes = document.querySelectorAll('.keyword-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    
    renderCustomKeywords();
    updateSelectedCount();
    
    // Reload graph without filters
    loadInitialGraph();
}

// Dropdown menu toggle
document.addEventListener('DOMContentLoaded', () => {
    const dropdown = document.querySelector('.dropdown');
    const toggle = document.querySelector('.dropdown-toggle');

    if (toggle && dropdown) {
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdown.classList.toggle('open');
        });

        // Close menu if clicking outside
        document.addEventListener('click', () => {
            dropdown.classList.remove('open');
        });
    }
});

// === Chat Bot Functionality ===
document.addEventListener('DOMContentLoaded', () => {
    const chatBalloon = document.getElementById('chat-bot-balloon');
    const chatWindow = document.getElementById('chat-bot-window');
    const chatClose = document.getElementById('chat-close');
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const chatMessages = document.getElementById('chat-messages');

    // Toggle chat window
    if (chatBalloon) {
        chatBalloon.addEventListener('click', () => {
            chatWindow.classList.toggle('active');
            if (chatWindow.classList.contains('active')) {
                chatInput.focus();
                // Add demo message if it's the first time opening
                if (!chatMessages.querySelector('.demo-message')) {
                    addDemoMessage();
                }
            }
        });
    }

    // Close chat window
    if (chatClose) {
        chatClose.addEventListener('click', (e) => {
            e.stopPropagation();
            chatWindow.classList.remove('active');
        });
    }

    // Send message function
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Remove empty state if it exists
        const emptyState = chatMessages.querySelector('.chat-empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        // Add user message
        addMessage(message, 'user');
        chatInput.value = '';
        
        // Show loading indicator
        const loadingMessage = addMessage('Lumi is thinking...', 'bot', false, 'loading');
        
        try {
            // Use the new homepage-chat endpoint
            const response = await fetch('/homepage-chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: message })
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Remove loading message
                const loadingMsg = chatMessages.querySelector('.loading');
                if (loadingMsg) {
                    loadingMsg.remove();
                }
                
                // Add response
                let responseText = data.answer;
                
                // Add retrieved articles info if available
                if (data.retrieved_articles && data.retrieved_articles.length > 0) {
                    const articleCount = data.retrieved_articles.length;
                    responseText += `\n\n📚 **${articleCount} relevant articles selected:**\n`;
                    
                    const articleNames = [];
                    // Show first 5 articles in detail, then summarize the rest
                    const articlesToShow = Math.min(5, data.retrieved_articles.length);
                    
                    for (let i = 0; i < articlesToShow; i++) {
                        const article = data.retrieved_articles[i];
                        const score = article.score || article.relevance_score || 0;
                        const scoreIcon = score > 0.6 ? '🟢' : score > 0.4 ? '🟡' : '🔴';
                        responseText += `${i + 1}. ${scoreIcon} ${article.article} (${score.toFixed(3)})\n`;
                        articleNames.push(article.article);
                    }
                    
                    // Add remaining articles to the list but don't display details
                    for (let i = articlesToShow; i < data.retrieved_articles.length; i++) {
                        articleNames.push(data.retrieved_articles[i].article);
                    }
                    
                    if (data.retrieved_articles.length > articlesToShow) {
                        responseText += `... and ${data.retrieved_articles.length - articlesToShow} more articles\n`;
                    }
                    
                    // Add button to open these articles in Ask Lumi
                    const articleNamesJson = JSON.stringify(articleNames).replace(/"/g, '&quot;');
                    responseText += `\n<button class="demo-button" onclick="loadRecommendedArticlesInLumi(${articleNamesJson})">
                        <i class="fas fa-rocket"></i> Open all ${articleCount} articles in Ask Lumi
                    </button>`;
                }
                
                addMessage(responseText, 'bot', true);
            } else {
                throw new Error('Chat service not available');
            }
        } catch (error) {
            console.log('Chat service error, using fallback response');
            
            // Remove loading message
            const loadingMsg = chatMessages.querySelector('.loading');
            if (loadingMsg) {
                loadingMsg.remove();
            }
            
            // Fallback response
            const fallbackResponses = [
                "Hello! I'm Lumi, your space research assistant. 🚀\n\nI'm currently learning about NASA experiments. You can explore the connections graph or visit the 'Ask Lumi' page for a more complete experience!",
                "Interesting question! 🌟\n\nFor detailed analysis, I recommend using the 'Ask Lumi' page where I can access a more complete database of scientific articles.",
                "I'm Lumi, space research specialist! 🛸\n\nFor specific questions about experiments, try the 'Ask Lumi' functionality which has access to more resources."
            ];
            
            const randomResponse = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
            addMessage(randomResponse, 'bot');
        }
    }

    // Add message to chat
    function addMessage(text, sender, isHtml = false, extraClass = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender} ${extraClass}`.trim();
        
        if (isHtml) {
            messageDiv.innerHTML = text;
        } else {
            messageDiv.textContent = text;
        }
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }


    // Add simple welcome message
    function addDemoMessage() {
        const demoHTML = `
            Hello! I'm Lumi, your space research assistant. 🚀<br><br>
            I can answer your questions about NASA experiments and space research. 
            For detailed analysis, visit the 'Ask Lumi' page!
        `;
        
        // Remove empty state
        const emptyState = chatMessages.querySelector('.chat-empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        addMessage(demoHTML, 'bot', true, 'demo-message');
    }

    // Send button click
    if (chatSend) {
        chatSend.addEventListener('click', sendMessage);
    }

    // Enter key to send
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

});

// Global function for loading recommended articles from chat bot
function loadRecommendedArticlesInLumi(articleNames) {
    // Create URL parameters with the recommended article names
    const params = new URLSearchParams({
        articles: articleNames.join('|'), // Use | as separator
        source: 'lumi_recommendation',
        message: `✨ ${articleNames.length} articles recommended by Lumi!`
    });

    // Redirect to ask-lumi with articles
    window.location.href = `/ask-lumi?${params.toString()}`;
}