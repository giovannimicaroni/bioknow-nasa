class BioKnowdes {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.articles = [];
        this.loadArticles();
        this.handleUrlParameters();
    }

    initializeElements() {
        this.articlesManager = document.getElementById('articlesManager');
        this.articlesList = document.getElementById('articlesList');
        this.articleCount = document.getElementById('articleCount');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.chatMessages = document.getElementById('chatMessages');
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    attachEventListeners() {
        // Chat events
        this.sendBtn.addEventListener('click', this.sendMessage.bind(this));
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Clear session
        this.clearBtn.addEventListener('click', this.clearSession.bind(this));
    }

    handleUrlParameters() {
        const urlParams = new URLSearchParams(window.location.search);
        const loaded = urlParams.get('loaded');
        const message = urlParams.get('message');
        const articles = urlParams.get('articles');
        const source = urlParams.get('source');
        
        // Handle legacy format with loaded count and message
        if (loaded && message) {
            this.showSuccessMessage(decodeURIComponent(message));
            // Clean URL
            window.history.replaceState({}, document.title, window.location.pathname);
        }
        
        // Handle new format with articles list
        if (articles) {
            const articleList = articles.split('|')
                .filter(a => a.trim())
                .map(article => {
                    // URL decode the article name (replace + with spaces and decode other chars)
                    let decoded = decodeURIComponent(article.replace(/\+/g, ' '));
                    // Remove .pdf extension if present
                    if (decoded.endsWith('.pdf')) {
                        decoded = decoded.slice(0, -4);
                    }
                    return decoded.trim();
                });
            const displayMessage = message ? decodeURIComponent(message) : 
                `✨ ${articleList.length} articles loaded${source === 'external_system' ? ' from external system' : ''}!`;
            
            this.loadArticlesByNames(articleList);
            this.showSuccessMessage(displayMessage);
            
            // Clean URL
            window.history.replaceState({}, document.title, window.location.pathname);
        }
    }

    showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(successDiv);
        
        setTimeout(() => {
            successDiv.classList.add('fade-out');
            setTimeout(() => successDiv.remove(), 500);
        }, 3000);
    }

    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.classList.add('fade-out');
            setTimeout(() => errorDiv.remove(), 500);
        }, 5000);
    }

    async loadArticles() {
        try {
            const response = await fetch('/ask-lumi/documents');
            const data = await response.json();
            
            if (data.documents) {
                this.articles = data.documents;
                this.renderArticles();
                this.updateInputState();
            }
        } catch (error) {
            console.error('Error loading articles:', error);
        }
    }

    async loadArticlesByNames(articleNames) {
        try {
            const response = await fetch('/ask-lumi/api/load-articles', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filenames: articleNames
                })
            });

            const result = await response.json();
            
            if (result.success) {
                // Reload articles from server to get updated list
                await this.loadArticles();
                console.log(`Successfully loaded ${result.loaded_count} articles`);
            } else {
                console.error('Error loading articles:', result.error);
                this.showErrorMessage(`Error loading articles: ${result.error}`);
            }
        } catch (error) {
            console.error('Error loading articles:', error);
            this.showErrorMessage(`Error loading articles: ${error.message}`);
        }
    }

    renderArticles() {
        const count = this.articles.length;
        this.articleCount.textContent = count;

        if (count === 0) {
            this.articlesList.innerHTML = `
                <div class="no-articles">
                    <i class="fas fa-search"></i>
                    <p>No articles loaded</p>
                    <a href="/articles" class="btn btn-primary btn-sm">
                        <i class="fas fa-plus"></i> Browse Articles
                    </a>
                </div>
            `;
            return;
        }

        this.articlesList.innerHTML = this.articles.map(article => `
            <div class="article-item" data-id="${article.id}">
                <div class="article-header">
                    <h4 class="article-title">${article.filename}</h4>
                    <div class="article-actions">
                        <button class="btn-icon eye-icon ${article.selected ? 'active' : 'inactive'}" 
                                onclick="bioKnow.toggleArticle('${article.id}')" 
                                title="${article.selected ? 'Remove from analysis' : 'Include in analysis'}">
                            <i class="fas ${article.selected ? 'fa-eye' : 'fa-eye-slash'}"></i>
                        </button>
                        ${this.getArticleButtons(article)}
                        <button class="btn-icon" onclick="bioKnow.removeArticle('${article.id}')" 
                                title="Remove article">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    getArticleButtons(article) {
        let buttons = '';
        
        // Full Text button (PMC link)
        if (article.pmc_link) {
            buttons += `<a href="${article.pmc_link}" target="_blank" class="btn-icon" title="Full Text (PMC)">
                <i class="fas fa-file-alt"></i>
            </a>`;
        }
        
        // Download PDF button
        if (article.download_url) {
            buttons += `<a href="${article.download_url}" target="_blank" class="btn-icon" title="Download PDF">
                <i class="fas fa-download"></i>
            </a>`;
        }
        
        return buttons;
    }

    async toggleArticle(articleId) {
        try {
            const response = await fetch(`/ask-lumi/documents/toggle/${articleId}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                await this.loadArticles();
            }
        } catch (error) {
            console.error('Error toggling article:', error);
        }
    }

    async removeArticle(articleId) {
        if (!confirm('Are you sure you want to remove this article?')) {
            return;
        }

        try {
            const response = await fetch(`/ask-lumi/documents/delete/${articleId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                await this.loadArticles();
            }
        } catch (error) {
            console.error('Error removing article:', error);
        }
    }

    updateInputState() {
        const hasSelectedArticles = this.articles.some(doc => doc.selected);
        this.messageInput.disabled = !hasSelectedArticles;
        this.sendBtn.disabled = !hasSelectedArticles;
        
        if (hasSelectedArticles) {
            this.messageInput.placeholder = "Ask questions about the selected articles...";
        } else {
            this.messageInput.placeholder = "Select articles to start asking questions...";
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        this.addMessage('user', message);
        this.messageInput.value = '';
        this.showLoading();

        try {
            const response = await fetch('/ask-lumi/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            
            if (data.response) {
                this.addMessage('assistant', data.response);
            } else if (data.error) {
                this.addMessage('assistant', `Error: ${data.error}`);
            } else {
                this.addMessage('assistant', 'Sorry, I encountered an error processing your request.');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('assistant', 'Sorry, I encountered an error processing your request.');
        } finally {
            this.hideLoading();
        }
    }

    addMessage(type, content) {
        const messageDiv = document.createElement('div');
        // User messages on the right, assistant on the left
        messageDiv.className = `message ${type === 'user' ? 'user-message' : 'assistant-message'}`;
        
        if (type === 'assistant') {
            const processedContent = this.processAIResponse(content);
            messageDiv.innerHTML = `
                <div class="message-header">
                    <i class="fas fa-robot"></i>
                    <span>Lumi</span>
                </div>
                <div class="message-content">${processedContent}</div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-header">
                    <i class="fas fa-user"></i>
                    <span>You</span>
                </div>
                <div class="message-content">${content}</div>
            `;
        }

        // Remove welcome message if present
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    processAIResponse(content) {
        // Extract thinking process if present
        const thinkingRegex = /<think>([\s\S]*?)<\/think>/g;
        let processedContent = content;
        let thinkingContent = '';

        const thinkingMatches = content.match(thinkingRegex);
        if (thinkingMatches) {
            thinkingContent = thinkingMatches.map(match => 
                match.replace(/<\/?think>/g, '')
            ).join('\n\n');
            
            processedContent = content.replace(thinkingRegex, '').trim();
        }

        let result = this.formatText(processedContent);

        if (thinkingContent) {
            result += `
                <div class="thinking-section">
                    <button class="thinking-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
                        <i class="fas fa-brain"></i> Show AI Thinking Process
                        <i class="fas fa-chevron-down"></i>
                    </button>
                    <div class="thinking-content hidden">
                        ${this.formatText(thinkingContent)}
                    </div>
                </div>
            `;
        }

        return result;
    }

    formatText(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
    }

    async clearSession() {
        if (!confirm('Are you sure you want to clear all articles and chat history?')) {
            return;
        }

        try {
            const response = await fetch('/ask-lumi/clear');
            if (response.ok) {
                this.articles = [];
                this.renderArticles();
                this.updateInputState();
                
                // Clear chat messages
                this.chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <i class="fas fa-robot"></i>
                        <h3>Welcome to BioKnowdes!</h3>
                        <p>Browse and load NASA scientific articles to start asking questions about space research.</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error clearing session:', error);
        }
    }

    showLoading() {
        // Instead of overlay, show typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message typing-indicator';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="message-header">
                <i class="fas fa-robot"></i>
                <span>Lumi</span>
            </div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        this.chatMessages.appendChild(typingDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    hideLoading() {
        // Remove typing indicator
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

// Initialize the application
let bioKnow;
document.addEventListener('DOMContentLoaded', () => {
    bioKnow = new BioKnowdes();
});