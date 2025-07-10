// Global variables
let isLoading = false;
let messageHistory = [];

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const toastContainer = document.getElementById('toastContainer');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    checkServerStatus();
    messageInput.focus();
    
    // Auto-resize message input
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
});

// Check server status
async function checkServerStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateStatus('online', 'Connected');
            showToast('Connected to Smart Travel Advisor', 'success');
        } else {
            updateStatus('offline', 'Service Unavailable');
            showToast('Service is currently unavailable', 'error');
        }
    } catch (error) {
        updateStatus('offline', 'Connection Failed');
        showToast('Failed to connect to server', 'error');
    }
}

// Update status indicator
function updateStatus(status, text) {
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = text;
}

// Handle key press in message input
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Add user message to chat
    addMessage(message, 'user');
    messageInput.value = '';
    
    // Show loading state
    setLoading(true);
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Add bot response to chat
        addBotResponse(data);
        
        // Store in history
        messageHistory.push({
            user: message,
            bot: data,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot', {
            error: true,
            route: 'ERROR'
        });
        showToast('Failed to get response from server', 'error');
    } finally {
        setLoading(false);
    }
}

// Send quick message
function sendQuickMessage(message) {
    messageInput.value = message;
    sendMessage();
}

// Add message to chat
function addMessage(text, sender, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.innerHTML = formatMessage(text);
    
    content.appendChild(messageText);
    
    // Add metadata info for bot messages
    if (sender === 'bot' && metadata.route) {
        const messageInfo = document.createElement('div');
        messageInfo.className = 'message-info';
        
        const routeBadge = document.createElement('span');
        routeBadge.className = `route-badge ${metadata.route.toLowerCase()}`;
        routeBadge.textContent = metadata.route;
        
        messageInfo.appendChild(routeBadge);
        
        if (metadata.confidence) {
            const confidenceSpan = document.createElement('span');
            confidenceSpan.textContent = `Confidence: ${metadata.confidence}`;
            messageInfo.appendChild(confidenceSpan);
        }
        
        content.appendChild(messageInfo);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add bot response with structured data
function addBotResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    
    if (data.route === 'ML' && data.result.predicted_price) {
        // ML Response with prediction
        messageText.innerHTML = formatMLResponse(data);
    } else if (data.route === 'RAG' && data.result.answer) {
        // RAG Response
        messageText.innerHTML = formatMessage(data.result.answer);
    } else if (data.result.error) {
        // Error Response
        messageText.innerHTML = `<p>❌ ${data.result.error}</p>`;
    } else {
        // Fallback
        messageText.innerHTML = '<p>I apologize, but I couldn\'t process your request properly.</p>';
    }
    
    content.appendChild(messageText);
    
    // Add metadata info
    const messageInfo = document.createElement('div');
    messageInfo.className = 'message-info';
    
    const routeBadge = document.createElement('span');
    routeBadge.className = `route-badge ${data.route.toLowerCase()}`;
    routeBadge.textContent = data.route;
    
    messageInfo.appendChild(routeBadge);
    
    if (data.intent_info && data.intent_info.reason) {
        const reasonSpan = document.createElement('span');
        reasonSpan.textContent = `Reason: ${data.intent_info.reason}`;
        messageInfo.appendChild(reasonSpan);
    }
    
    content.appendChild(messageInfo);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Format ML response with prediction details
function formatMLResponse(data) {
    const result = data.result;
    let html = '<p>✈️ <strong>Flight Price Prediction</strong></p>';
    
    if (result.predicted_price) {
        html += `
            <div class="prediction-result">
                <div class="price-display">₹${result.predicted_price.toLocaleString()}</div>
                <div class="prediction-details">
                    <div class="detail-item">
                        <div class="detail-label">Currency</div>
                        <div class="detail-value">${result.currency || 'INR'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Confidence</div>
                        <div class="detail-value">${result.confidence || 'Medium'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Type</div>
                        <div class="detail-value">${result.prediction_type || 'Flight Price'}</div>
                    </div>
                </div>
            </div>
        `;
        
        if (result.features) {
            html += '<p><strong>Based on:</strong></p><ul>';
            Object.entries(result.features).forEach(([key, value]) => {
                if (value && value !== 'Unknown') {
                    html += `<li><strong>${key.replace(/_/g, ' ')}:</strong> ${value}</li>`;
                }
            });
            html += '</ul>';
        }
    }
    
    return html;
}

// Format message text
function formatMessage(text) {
    // Convert markdown-like formatting
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert line breaks
    text = text.replace(/\n/g, '<br>');
    
    return `<p>${text}</p>`;
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    sendButton.disabled = loading;
    
    if (loading) {
        loadingOverlay.classList.add('show');
        sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    } else {
        loadingOverlay.classList.remove('show');
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    toastContainer.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Export chat history
function exportChatHistory() {
    const dataStr = JSON.stringify(messageHistory, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `chat_history_${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

// Clear chat history
function clearChatHistory() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">
                        <p>Hello! I'm your Smart Travel Advisor. I can help you with:</p>
                        <ul>
                            <li>✈️ Flight price predictions</li>
                            <li>📋 Travel policies and procedures</li>
                            <li>📄 Documentation requirements</li>
                            <li>💡 General travel information</li>
                        </ul>
                        <p>What would you like to know?</p>
                    </div>
                </div>
            </div>
        `;
        messageHistory = [];
        showToast('Chat history cleared', 'success');
    }
}

// Periodic health check
setInterval(checkServerStatus, 30000); // Check every 30 seconds
