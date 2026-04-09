const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Using the deployed Render URL as requested
const API_URL = 'https://naija-rights.onrender.com/chat';

let conversationHistory = [];

userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? '👤' : '⚖️';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    
    if (role === 'ai') {
        // Parse markdown for AI responses
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.textContent = content;
    }
    
    // Append to message div based on layout
    if (role === 'user') {
        msgDiv.appendChild(contentDiv);
        msgDiv.appendChild(avatar);
    } else {
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
    }
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showLoading() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai-message loading-indicator';
    msgDiv.innerHTML = `
        <div class="avatar">⚖️</div>
        <div class="content loading-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    
    // Disable input
    userInput.value = '';
    userInput.style.height = 'auto';
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message to UI
    addMessage('user', text);
    
    // Show AI loading
    const loadingEl = showLoading();
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: text,
                history: conversationHistory,
                eli15: false
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server err: ${response.status}`);
        }
        
        const data = await response.json();
        const answer = data.answer;
        
        // Remove loading
        loadingEl.remove();
        
        // Add AI response
        addMessage('ai', answer);
        
        // Update history
        conversationHistory.push({ role: 'user', content: text });
        conversationHistory.push({ role: 'assistant', content: answer });
        
        // Keep history manageable
        if(conversationHistory.length > 10) conversationHistory = conversationHistory.slice(-10);
        
    } catch (error) {
        console.error('API Error:', error);
        loadingEl.remove();
        addMessage('ai', "Sorry boss, my network just shake small. Try again make we see.");
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}
