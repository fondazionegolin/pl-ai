// RAG (Retrieval-Augmented Generation) page functionality

// Wait for window load to ensure all resources are fully loaded
window.addEventListener('load', function() {
    console.log('RAG page fully loaded');
    
    // Initialize variables
    let uploadedFiles = [];
    let processingStatus = false;
    let knowledgeBaseReady = false;
    
    // Function to safely get DOM elements
    function getElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with ID '${id}' not found`);
        }
        return element;
    }
    
    // Function to safely add event listeners
    function addSafeEventListener(element, event, handler) {
        if (element) {
            element.addEventListener(event, handler);
            return true;
        }
        return false;
    }
    
    // Get all required elements
    const elements = {
        // Processing status elements
        processingStatus: getElement('processing-status'),
        processDetails: getElement('process-details'),
        currentProcess: getElement('current-process'),
        progressBar: getElement('progress-bar'),
        progressPercentage: getElement('progress-percentage'),
        
        // File upload elements
        fileInput: getElement('fallback-file-input'),
        browseButton: getElement('browse-files-btn'),
        dropzoneContainer: getElement('dropzone-container'),
        processButton: getElement('process-documents-btn'),
        
        // Chat elements
        chatForm: getElement('chat-form'),
        chatInput: getElement('chat-input'),
        chatMessages: getElement('chat-messages'),
        
        // Knowledge base elements
        knowledgeBaseStatus: getElement('knowledge-base-status'),
        statusMessage: getElement('status-message'),
        statusDetails: getElement('status-details'),
        statusIcon: getElement('status-icon'),
        documentList: getElement('document-list'),
        clearKnowledgeBaseBtn: getElement('clear-knowledge-base-btn')
    };
    
    // Check if we have the minimum required elements to function
    const hasUploadElements = elements.fileInput && elements.browseButton && 
                             elements.dropzoneContainer && elements.processButton;
    
    const hasProcessingElements = elements.processingStatus && elements.processDetails && 
                                elements.currentProcess && elements.progressBar && 
                                elements.progressPercentage;
    
    // Function to update processing status
    function updateProcessingStatus(message, isError = false) {
        console.log('Updating processing status:', message, isError);
        
        if (elements.processingStatus) {
            elements.processingStatus.classList.remove('hidden');
        }
        
        if (elements.currentProcess) {
            elements.currentProcess.className = isError ? 
                'text-xs font-semibold inline-block text-red-600' : 
                'text-xs font-semibold inline-block text-indigo-600';
            elements.currentProcess.textContent = isError ? `Errore: ${message}` : message;
        }
        
        if (elements.processDetails) {
            const timestamp = new Date().toLocaleTimeString();
            const detailElement = document.createElement('p');
            detailElement.className = isError ? 'text-sm text-red-600' : 'text-sm text-gray-600';
            detailElement.textContent = `[${timestamp}] ${message}`;
            elements.processDetails.appendChild(detailElement);
            elements.processDetails.scrollTop = elements.processDetails.scrollHeight;
        }
    }
    
    // Function to update progress bar
    function updateProgressBar(percentage) {
        if (elements.progressBar && elements.progressPercentage) {
            elements.progressBar.style.width = `${percentage}%`;
            elements.progressPercentage.textContent = `${percentage}%`;
        }
    }
    
    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Function to update the file list display
    function updateFileList() {
        if (!elements.documentList) return;
        
        elements.documentList.innerHTML = '';
        
        if (uploadedFiles.length === 0) {
            elements.documentList.innerHTML = '<p class="text-gray-500 text-center">Nessun file selezionato</p>';
            return;
        }
        
        uploadedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'flex items-center justify-between p-2 border-b';
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'flex items-center';
            
            const icon = document.createElement('span');
            icon.className = 'text-indigo-500 mr-2';
            icon.innerHTML = '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"></path></svg>';
            
            const fileName = document.createElement('span');
            fileName.className = 'text-sm font-medium';
            fileName.textContent = file.name;
            
            const fileSize = document.createElement('span');
            fileSize.className = 'text-xs text-gray-500 ml-2';
            fileSize.textContent = formatFileSize(file.size);
            
            fileInfo.appendChild(icon);
            fileInfo.appendChild(fileName);
            fileInfo.appendChild(fileSize);
            
            const removeButton = document.createElement('button');
            removeButton.className = 'text-red-500 hover:text-red-700';
            removeButton.innerHTML = '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>';
            removeButton.addEventListener('click', () => {
                uploadedFiles.splice(index, 1);
                updateFileList();
                if (uploadedFiles.length === 0 && elements.processButton) {
                    elements.processButton.disabled = true;
                }
            });
            
            fileItem.appendChild(fileInfo);
            fileItem.appendChild(removeButton);
            elements.documentList.appendChild(fileItem);
        });
    }
    
    // Function to process documents
    async function processDocuments(event) {
        // Prevent default form submission if called from a form
        if (event) {
            event.preventDefault();
        }
        
            if (uploadedFiles.length === 0) {
            updateProcessingStatus("Nessun file da processare", true);
                return;
            }
            
        if (processingStatus) {
                updateProcessingStatus("Elaborazione già in corso", true);
                return;
            }
            
        processingStatus = true;
            updateProcessingStatus("Avvio elaborazione documenti...");
            updateProgressBar(5);
            
        try {
            console.log("[DEBUG] Starting document processing");
            // Upload files one by one
            const uploadedFileIds = [];
            for (const file of uploadedFiles) {
                console.log(`[DEBUG] Uploading file: ${file.name}`);
            const formData = new FormData();
                formData.append('file', file);
            
                const uploadResponse = await fetch('/upload_rag_document', {
                method: 'POST',
                body: formData
                });
                
                if (!uploadResponse.ok) {
                    const errorData = await uploadResponse.json();
                    throw new Error(`Errore durante il caricamento del file ${file.name}: ${errorData.error || 'Errore sconosciuto'}`);
                }
                
                const uploadResult = await uploadResponse.json();
                console.log(`[DEBUG] Upload result for ${file.name}:`, uploadResult);
                if (uploadResult.success) {
                    uploadedFileIds.push(uploadResult.file_id);
                }
            }
            
            updateProgressBar(20);
            updateProcessingStatus("File caricati con successo");
            
            console.log("[DEBUG] Starting text extraction");
            // Start text extraction
            const extractionResponse = await fetch('/process_rag_documents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
                body: JSON.stringify({ file_ids: uploadedFileIds })
            });
            
            if (!extractionResponse.ok) {
                const errorData = await extractionResponse.json();
                throw new Error(`Errore durante l'estrazione del testo: ${errorData.error || 'Errore sconosciuto'}`);
            }
            
            updateProgressBar(40);
            updateProcessingStatus("Estrazione testo in corso...");
            
            // Poll for text extraction status
            let extractionComplete = false;
            let retryCount = 0;
            const maxRetries = 60; // 1 minute timeout
            
            while (!extractionComplete && retryCount < maxRetries) {
                console.log("[DEBUG] Checking text extraction status");
                const statusResponse = await fetch('/check_text_extraction_status');
                const status = await statusResponse.json();
                
                console.log("[DEBUG] Extraction status:", status);
                
                if (status.error) {
                    throw new Error(status.error);
                }
                
                if (status.completed) {
                    extractionComplete = true;
                    updateProgressBar(60);
                    updateProcessingStatus("Testo estratto con successo");
                    break; // Exit the loop when complete
                } else {
                    retryCount++;
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            if (!extractionComplete) {
                throw new Error("Timeout durante l'estrazione del testo");
            }
            
            // Create embeddings
            updateProcessingStatus("Creazione embeddings in corso...");
            console.log("[DEBUG] Starting embedding creation");
            const embeddingResponse = await fetch('/create_rag_embeddings', {
                method: 'POST'
            });
            
            if (!embeddingResponse.ok) {
                const errorData = await embeddingResponse.json();
                throw new Error(`Errore durante la creazione degli embeddings: ${errorData.error || 'Errore sconosciuto'}`);
            }
            
            updateProgressBar(80);
            
            // Poll for embedding status
            let embeddingComplete = false;
            retryCount = 0;
            
            while (!embeddingComplete && retryCount < maxRetries) {
                console.log("[DEBUG] Checking embedding status");
                const statusResponse = await fetch('/check_embedding_status');
                const status = await statusResponse.json();
                
                console.log("[DEBUG] Embedding status:", status);
                
                if (status.error) {
                    throw new Error(status.error);
                }
                
                if (status.completed) {
                    embeddingComplete = true;
                    updateProgressBar(100);
                    updateProcessingStatus("Knowledge base creata con successo!");
                    knowledgeBaseReady = true;
                    
                    if (elements.chatInput && elements.chatForm) {
                        elements.chatInput.disabled = false;
                        elements.chatForm.querySelector('button').disabled = false;
                    }
                    
                    // Update knowledge base status
                    updateKnowledgeBaseStatus(true);
                    break; // Exit the loop when complete
                } else {
                    retryCount++;
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            if (!embeddingComplete) {
                throw new Error("Timeout durante la creazione degli embeddings");
            }
            
        } catch (error) {
            console.error('[ERROR] Error processing documents:', error);
            updateProcessingStatus(error.message, true);
        } finally {
            processingStatus = false;
            uploadedFiles = [];
            updateFileList();
            if (elements.processButton) {
                elements.processButton.disabled = true;
            }
        }
    }
    
    // Function to update knowledge base status
    function updateKnowledgeBaseStatus(isReady) {
        if (!elements.knowledgeBaseStatus || !elements.statusMessage || !elements.statusDetails) return;
        
        elements.knowledgeBaseStatus.classList.remove('hidden');
        
        if (isReady) {
            elements.statusMessage.textContent = 'Knowledge base pronta';
            elements.statusDetails.textContent = 'Puoi iniziare a chattare con il bot';
            elements.statusIcon.innerHTML = '<svg class="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
            
            if (elements.clearKnowledgeBaseBtn) {
                elements.clearKnowledgeBaseBtn.disabled = false;
            }
        } else {
            elements.statusMessage.textContent = 'Nessun documento processato';
            elements.statusDetails.textContent = 'Carica e processa i documenti per creare la knowledge base';
            elements.statusIcon.innerHTML = '<svg class="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
            
            if (elements.clearKnowledgeBaseBtn) {
                elements.clearKnowledgeBaseBtn.disabled = true;
            }
        }
    }
    
    // Function to clear knowledge base
    async function clearKnowledgeBase() {
        if (!confirm('Sei sicuro di voler cancellare la knowledge base?')) return;
        
        try {
            const response = await fetch('/clear_rag_knowledge_base', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Errore durante la cancellazione della knowledge base');
            }
            
                knowledgeBaseReady = false;
                updateKnowledgeBaseStatus(false);
                
            if (elements.chatInput && elements.chatForm) {
                elements.chatInput.disabled = true;
                elements.chatForm.querySelector('button').disabled = true;
            }
            
            updateProcessingStatus("Knowledge base cancellata con successo");
            
        } catch (error) {
            console.error('Error clearing knowledge base:', error);
            updateProcessingStatus(error.message, true);
        }
    }
    
    // Gestione della chat
    function setupChat() {
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('chat-input');
        const chatMessages = document.getElementById('chat-messages');
        const sendButton = document.getElementById('send-button') || chatForm?.querySelector('button[type="submit"]');
        
        if (!chatForm || !chatInput || !chatMessages) {
            console.error('Chat elements not found:', {
                chatForm: !!chatForm,
                chatInput: !!chatInput,
                chatMessages: !!chatMessages,
                sendButton: !!sendButton
            });
            return;
        }

        /* Function to add a message to the chat interface */
        function addMessage(content, isUser = false) {
            const messagesContainer = elements.chatMessages;
            if (!messagesContainer) return;

            const messageElement = document.createElement('div');
            messageElement.className = `flex w-full mt-2 space-x-3 ${isUser ? 'ml-auto justify-end' : 'justify-start'}`;

            if (!isUser) {
                 // Bot avatar placeholder
                 const botAvatar = document.createElement('div');
                 botAvatar.className = 'flex-shrink-0 h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center';
                 botAvatar.innerHTML = '<svg class="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a6 6 0 00-6 6v3.586l1.707 1.707A2 2 0 016.414 14H15.586a2 2 0 011.707-.586L18 11.586V8a6 6 0 00-6-6zM10 16a2 2 0 110-4 2 2 0 010 4z"></path></svg>';
                 messageElement.appendChild(botAvatar);
            }

            const messageBubble = document.createElement('div');
            messageBubble.className = `flex-shrink-0 rounded-xl p-3 ${isUser ? 'bg-blue-600 text-white rounded-br-none' : 'bg-gray-200 text-gray-800 rounded-bl-none'} max-w-[80%]`;
            messageBubble.style.wordBreak = 'break-word';
            messageBubble.style.overflowWrap = 'break-word';

            const messageText = document.createElement('p');
            messageText.className = 'text-sm whitespace-pre-wrap'; // Keep whitespace like newlines
            messageText.textContent = content;

            messageBubble.appendChild(messageText);
            messageElement.appendChild(messageBubble);

            if (isUser) {
                 // User avatar placeholder (optional, could add user icon)
                 const userAvatar = document.createElement('div');
                 userAvatar.className = 'flex-shrink-0 h-10 w-10 rounded-full bg-blue-300 flex items-center justify-center';
                 userAvatar.innerHTML = '<svg class="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"></path></svg>';
                 messageElement.appendChild(userAvatar);
            }

            messagesContainer.appendChild(messageElement);
            messagesContainer.scrollTop = messagesContainer.scrollHeight; // Auto-scroll to latest message
            return messageElement; // Return the message element for potential future updates
        }

        /* Function to send a chat message */
        async function sendChatMessage(message) {
            if (!knowledgeBaseReady) {
                alert('La knowledge base non è ancora pronta. Attendere il completamento dell\'elaborazione dei documenti.');
                return;
            }

            addMessage(message, true); // Add user message to UI
            elements.chatInput.value = ''; // Clear input
            elements.chatInput.disabled = true; // Disable input while thinking

            // Add a placeholder message for the bot's response with a loading indicator
            const botMessageElement = addMessage('...', false);
            const loadingIndicator = document.createElement('span');
            loadingIndicator.className = 'ml-2 inline-block animate-pulse';
            loadingIndicator.textContent = '🧠'; // Thinking emoji or similar
            botMessageElement.querySelector('p').appendChild(loadingIndicator);

            try {
                const response = await fetch('/rag_chat', {
            method: 'POST',
            headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/event-stream' // Indicate expecting a stream
            },
            body: JSON.stringify({ message: message })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`Errore nella chat: ${errorData.error || 'Errore sconosciuto'}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let receivedText = '';
                
                // Update the bot message element directly
                const messageTextElement = botMessageElement.querySelector('p');
                messageTextElement.textContent = ''; // Clear the '...' and loading indicator

                // --- DEBUG: Check applied style ---
                console.log("[DEBUG] messageTextElement white-space style:", window.getComputedStyle(messageTextElement).whiteSpace);
                // --- End Debug ---

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    // Assuming the backend sends plain text chunks
                    const chunk = decoder.decode(value, { stream: true });
                    
                    // --- DEBUG: Log received chunk ---
                    console.log("[DEBUG] Received chunk:", chunk);
                    // --- End Debug ---

                    // Append the new chunk to the displayed text
                    // Using textContent += chunk to append text and rely on whitespace-pre-wrap
                    messageTextElement.textContent += chunk; 
                    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight; // Auto-scroll
                }

            } catch (error) {
                console.error('Error sending chat message:', error);
                // Replace loading indicator with error message
                const messageTextElement = botMessageElement.querySelector('p');
                messageTextElement.textContent = `Errore: ${error.message}`; // Display error in the message bubble
            } finally {
                elements.chatInput.disabled = false; // Re-enable input
            }
        }

        /* Setup chat form submission */
        if (elements.chatForm && elements.chatInput) {
            elements.chatForm.addEventListener('submit', function(e) {
                e.preventDefault();
                const message = elements.chatInput.value.trim();
                if (message) {
                    sendChatMessage(message);
                }
            });
        }

        // Gestione del click sul pulsante di invio (se esiste)
        if (sendButton) {
            sendButton.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                const message = chatInput.value.trim();
                if (message) {
                    sendChatMessage(message);
                }
                return false;
            });
        }

        // Gestione dell'invio con tasto Enter
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                const message = chatInput.value.trim();
                if (message) {
                    sendChatMessage(message);
                }
                return false;
            }
        });

        // Abilita la chat se la knowledge base è pronta
        if (knowledgeBaseReady) {
            chatInput.disabled = false;
            if (sendButton) {
                sendButton.disabled = false;
            }
        } else {
            chatInput.disabled = true;
            if (sendButton) {
                sendButton.disabled = true;
            }
        }
    }
    
    // Initialize the UI if we have the required elements
    if (hasUploadElements && hasProcessingElements) {
        // Initial status update
        updateProcessingStatus('Sistema pronto per il caricamento dei documenti');
        
        // Set up event listeners for file upload
        addSafeEventListener(elements.browseButton, 'click', () => elements.fileInput.click());
        
        addSafeEventListener(elements.fileInput, 'change', (e) => {
            const files = e.target.files;
            if (files.length > 0) {
                uploadedFiles = Array.from(files);
                updateFileList();
                elements.processButton.disabled = false;
            }
        });
        
        addSafeEventListener(elements.dropzoneContainer, 'dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropzoneContainer.classList.add('border-indigo-500');
        });
        
        addSafeEventListener(elements.dropzoneContainer, 'dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropzoneContainer.classList.remove('border-indigo-500');
        });
        
        addSafeEventListener(elements.dropzoneContainer, 'drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropzoneContainer.classList.remove('border-indigo-500');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadedFiles = Array.from(files);
                updateFileList();
                elements.processButton.disabled = false;
            }
        });
        
        // Process Documents Button
        addSafeEventListener(elements.processButton, 'click', processDocuments);
        
        // Clear Knowledge Base Button
        addSafeEventListener(elements.clearKnowledgeBaseBtn, 'click', clearKnowledgeBase);
        
        // Chat Form
        setupChat();
        
        // Check initial knowledge base status
    fetch('/check_rag_knowledge_base')
    .then(response => response.json())
    .then(data => {
                if (data.has_documents) {
                    knowledgeBaseReady = true;
                    updateKnowledgeBaseStatus(true);
                    setupChat(); // Reinizializza la chat quando la knowledge base è pronta
        }
    })
    .catch(error => {
        console.error('Error checking knowledge base status:', error);
                updateProcessingStatus('Errore durante il controllo dello stato della knowledge base', true);
    });
    } else {
        console.error('Required elements not found, RAG functionality may not work correctly');
        updateProcessingStatus('Errore di inizializzazione: elementi mancanti', true);
    }
});
