// Global variables
let preloadedBlobUrl = null;
let allKeywords = [];
let selectedKeywords = new Set();

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
    countBadge.textContent = `${selectedKeywords.size} selected`;
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
    
    if (query && query !== 'N/A') {
        params.append('query', query);
    }
    
    selectedKeywords.forEach(keyword => {
        params.append('filter_keywords[]', keyword);
    });
    
    if (params.toString()) {
        url += '?' + params.toString();
    }
    
    console.log('Applying filters:', url);
    iframe.src = url;
}

function clearFilters() {
    // Uncheck all checkboxes
    selectedKeywords.clear();
    
    const checkboxes = document.querySelectorAll('.keyword-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    
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
    function sendMessage() {
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
        // INTEGRAR O CHATBOT AQUI
        // Simulate bot response (replace with actual API call)
        setTimeout(() => {
            addMessage('Thanks for your message! I\'m still learning. Check back soon for full chat capabilities!', 'bot');
        }, 1000);
    }

    // Add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
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