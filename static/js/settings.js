class SettingsManager {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.loadSettings();
    }

    initializeElements() {
        this.form = document.getElementById('settingsForm');
        this.providerRadios = document.querySelectorAll('input[name="provider"]');
        this.temperatureSlider = document.getElementById('temperature');
        this.temperatureValue = document.querySelector('.range-value');
        this.testConnectionBtn = document.getElementById('testConnection');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.resetBtn = document.getElementById('resetDefaults');
        this.statusMessage = document.getElementById('statusMessage');
    }

    attachEventListeners() {
        // Provider selection
        this.providerRadios.forEach(radio => {
            radio.addEventListener('change', this.handleProviderChange.bind(this));
        });

        // Temperature slider
        this.temperatureSlider.addEventListener('input', (e) => {
            this.temperatureValue.textContent = e.target.value;
        });

        // Test connection
        this.testConnectionBtn.addEventListener('click', this.testConnection.bind(this));

        // Form submission
        this.form.addEventListener('submit', this.saveSettings.bind(this));

        // Reset defaults
        this.resetBtn.addEventListener('click', this.resetDefaults.bind(this));
    }

    handleProviderChange(e) {
        // Update UI based on selected provider
        const selectedProvider = e.target.value;
        console.log('Provider changed to:', selectedProvider);
    }

    async testConnection() {
        const localUrl = document.getElementById('local-url').value;
        const localModel = document.getElementById('local-model').value;

        if (!localUrl.trim()) {
            this.showConnectionStatus('error', 'URL cannot be empty');
            return;
        }

        this.showConnectionStatus('testing', 'Testing connection...');
        this.testConnectionBtn.disabled = true;

        try {
            // Test models endpoint first
            const modelsUrl = localUrl.replace('/chat/completions', '/models');
            const response = await fetch('/ask-lumi/api/test-connection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: localUrl,
                    model: localModel
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showConnectionStatus('success', '✓ Connection successful!');
            } else {
                this.showConnectionStatus('error', `✗ ${result.error}`);
            }
        } catch (error) {
            this.showConnectionStatus('error', `✗ Error: ${error.message}`);
        } finally {
            this.testConnectionBtn.disabled = false;
        }
    }

    showConnectionStatus(type, message) {
        this.connectionStatus.className = `connection-status ${type}`;
        this.connectionStatus.textContent = message;
    }

    async saveSettings(e) {
        e.preventDefault();

        const settings = this.gatherSettings();
        
        try {
            const response = await fetch('/ask-lumi/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            const result = await response.json();

            if (result.success) {
                this.showStatusMessage('success', '✓ Settings saved successfully!');
                // Redirect back to chat page after 2 seconds
                setTimeout(() => {
                    window.location.href = '/ask-lumi';
                }, 2000);
            } else {
                this.showStatusMessage('error', `✗ Error saving: ${result.error}`);
            }
        } catch (error) {
            this.showStatusMessage('error', `✗ Error: ${error.message}`);
        }
    }

    gatherSettings() {
        const selectedProvider = document.querySelector('input[name="provider"]:checked').value;
        
        const settings = {
            provider: selectedProvider,
            temperature: parseFloat(this.temperatureSlider.value),
            maxTokens: parseInt(document.getElementById('max-tokens').value)
        };

        // Add provider-specific settings
        switch (selectedProvider) {
            case 'openai':
                settings.openai = {
                    apiKey: document.getElementById('openai-key').value,
                    model: document.getElementById('openai-model').value
                };
                break;
            
            case 'anthropic':
                settings.anthropic = {
                    apiKey: document.getElementById('anthropic-key').value,
                    model: document.getElementById('anthropic-model').value
                };
                break;
            
            case 'gemini':
                settings.gemini = {
                    apiKey: document.getElementById('gemini-key').value,
                    model: document.getElementById('gemini-model').value
                };
                break;
            
            case 'local':
                settings.local = {
                    url: document.getElementById('local-url').value,
                    model: document.getElementById('local-model').value
                };
                break;
        }

        return settings;
    }

    async loadSettings() {
        try {
            const response = await fetch('/ask-lumi/api/settings');
            const settings = await response.json();

            if (settings) {
                this.applySettings(settings);
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    applySettings(settings) {
        // Set provider
        if (settings.provider) {
            const providerRadio = document.getElementById(settings.provider);
            if (providerRadio) {
                providerRadio.checked = true;
            }
        }

        // Set common settings
        if (settings.temperature !== undefined) {
            this.temperatureSlider.value = settings.temperature;
            this.temperatureValue.textContent = settings.temperature;
        }

        if (settings.maxTokens !== undefined) {
            document.getElementById('max-tokens').value = settings.maxTokens;
        }

        // Set provider-specific settings
        if (settings.openai) {
            if (settings.openai.apiKey) {
                document.getElementById('openai-key').value = settings.openai.apiKey;
            }
            if (settings.openai.model) {
                document.getElementById('openai-model').value = settings.openai.model;
            }
        }

        if (settings.anthropic) {
            if (settings.anthropic.apiKey) {
                document.getElementById('anthropic-key').value = settings.anthropic.apiKey;
            }
            if (settings.anthropic.model) {
                document.getElementById('anthropic-model').value = settings.anthropic.model;
            }
        }

        if (settings.gemini) {
            if (settings.gemini.apiKey) {
                document.getElementById('gemini-key').value = settings.gemini.apiKey;
            }
            if (settings.gemini.model) {
                document.getElementById('gemini-model').value = settings.gemini.model;
            }
        }

        if (settings.local) {
            if (settings.local.url) {
                document.getElementById('local-url').value = settings.local.url;
            }
            if (settings.local.model) {
                document.getElementById('local-model').value = settings.local.model;
            }
        }
    }

    resetDefaults() {
        if (confirm('Are you sure you want to restore default settings?')) {
            // Reset to default values
            document.getElementById('local').checked = true;
            document.getElementById('local-url').value = 'http://172.28.160.1:3000';
            document.getElementById('local-model').value = 'deepseek/deepseek-r1-0528-qwen3-8b';
            this.temperatureSlider.value = 0.7;
            this.temperatureValue.textContent = '0.7';
            document.getElementById('max-tokens').value = 1000;

            // Clear API keys
            document.getElementById('openai-key').value = '';
            document.getElementById('anthropic-key').value = '';
            document.getElementById('gemini-key').value = '';

            this.showStatusMessage('success', '✓ Default settings restored!');
        }
    }

    showStatusMessage(type, message) {
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.textContent = message;
        this.statusMessage.classList.remove('hidden');

        // Hide after 5 seconds
        setTimeout(() => {
            this.statusMessage.classList.add('hidden');
        }, 5000);
    }
}

// Initialize settings manager
document.addEventListener('DOMContentLoaded', () => {
    new SettingsManager();
});