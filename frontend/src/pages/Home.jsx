import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import authStore from '../stores/authStore';

const ConversationSidebar = ({ conversations, selectedId, onSelect, onLogout, onNewConversation, onDelete }) => (
  <div className="w-64 h-full bg-gray-800 border-r border-gray-700 flex flex-col">
    <div className="p-4 border-b border-gray-700 flex justify-between items-center">
      <h1 className="text-white text-xl font-bold">ChatApp</h1>
      <button
        onClick={onLogout}
        className="text-gray-400 hover:text-gray-200 text-sm"
        title="Logout"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
        </svg>
      </button>
    </div>
    
    <div className="p-2">
      <button
        onClick={onNewConversation}
        className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 text-white rounded-md flex items-center justify-center text-sm transition-colors"
      >
        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
        New Conversation
      </button>
    </div>
    
    <nav className="flex-1 overflow-y-auto p-2" aria-label="Chat history">
      {conversations.map((convo) => (
        <div
          key={convo.id}
          className={`group relative mb-1 rounded cursor-pointer transition-colors
            ${selectedId === convo.id 
              ? 'bg-gray-700' 
              : 'hover:bg-gray-700'}`}
        >
          <button
            onClick={() => onSelect(convo.id)}
            className="w-full p-2 text-sm text-left text-white pr-8"
          >
            <div className="flex justify-between items-center">
              <span className="truncate flex-1 mr-2">{convo.title}</span>
              <span className="text-xs text-gray-400 whitespace-nowrap">
                {convo.message_count} messages
              </span>
            </div>
          </button>
          
          {/* Delete button - only visible on hover */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(convo.id);
            }}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
            title="Delete conversation"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      ))}
    </nav>
  </div>
);

const MessageBubble = ({ message }) => {
  const isUser = message.sender === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`w-[70%] p-2.5 rounded-lg text-sm border ${
        isUser 
          ? 'bg-blue-600 text-white border-blue-500' 
          : 'bg-gray-700 text-gray-100 border-gray-600'
      }`}>
        <p className="break-words whitespace-pre-wrap leading-snug">
          {message.text}
        </p>
      </div>
    </div>
  );
};

// Add CSS variables for easier theming
const mainStyles = `
  bg-gray-950
  text-gray-100
  [color-scheme:dark]
  h-screen
  overflow-hidden
`;

export default function Home() {
  const { logout } = useAuth();
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef(null);
  const containerRef = useRef(null);
  const [conversations, setConversations] = useState([]);
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const token = authStore(state => state.token);
  const [isRagEnabled, setIsRagEnabled] = useState(true);

  // Add sidebar toggle functionality
  const toggleSidebar = useCallback(() => setIsSidebarOpen(prev => !prev), []);
  
  // Add escape key handler
  const handleEscape = useCallback((e) => {
    if (e.key === 'Escape' && isSidebarOpen) {
      setIsSidebarOpen(false);
    }
  }, [isSidebarOpen]);

  // Add escape key listener
  useEffect(() => {
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [handleEscape]);

  // Auto-resize textarea effect
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const adjustHeight = () => {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    };

    adjustHeight();
    // Add resize listener for window changes
    window.addEventListener('resize', adjustHeight);
    return () => window.removeEventListener('resize', adjustHeight);
  }, [newMessage]);

  // Add scroll effect
  useEffect(() => {
    if (containerRef.current) {
        containerRef.current.scrollTo({
            top: containerRef.current.scrollHeight,
            behavior: 'smooth'
        });
    }
  }, [messages]);

  // Fetch conversations on mount
  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const response = await fetch('/api/textgen/conversations', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        if (!response.ok) throw new Error('Failed to fetch conversations');
        const data = await response.json();
        setConversations(data);
      } catch (error) {
        console.error('Error fetching conversations:', error);
      }
    };
    
    fetchConversations();
  }, [token]);

  // Update handleSendMessage to include full conversation history
  const handleSendMessage = async (e) => {
    e.preventDefault();
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || isSending) return;

    const userMessage = {
      id: crypto.randomUUID(),
      text: trimmedMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    const loadingMessage = {
      id: `loading-${Date.now()}`,
      text: '...',
      sender: 'assistant',
      isTemp: true
    };

    try {
      setIsSending(true);
      setMessages(prev => [...prev, userMessage, loadingMessage]);
      setNewMessage('');

      // Convert message history to the format expected by the API
      const messageHistory = messages.map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));

      // Add the current message
      messageHistory.push({ role: 'user', content: trimmedMessage });

      // Update API call to include full message history
      const response = await fetch('/api/textgen/generate', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({
          messages: messageHistory,
          conversation_id: selectedConversationId,
          use_rag: isRagEnabled
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'API request failed');
      }

      const data = await response.json();

      // Validate the response
      if (!data.content || typeof data.content !== 'string') {
        throw new Error('Invalid response format from server');
      }

      // Update messages with response data
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingMessage.id),
        {
          id: data.message_id,
          text: data.content,
          sender: 'assistant',
          timestamp: new Date().toISOString()
        }
      ]);

      // Refresh conversations list
      if (!selectedConversationId) {
        const convResponse = await fetch('/api/textgen/conversations', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        if (!convResponse.ok) {
          throw new Error('Failed to refresh conversations');
        }
        const convData = await convResponse.json();
        setConversations(convData);
        setSelectedConversationId(data.conversation_id);
      }

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingMessage.id),
        {
          id: crypto.randomUUID(),
          text: `Error: ${error.message}. Please try again or contact support if the issue persists.`,
          sender: 'assistant',
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setIsSending(false);
    }
  };

  // Add handler for conversation selection
  const handleSelectConversation = async (convoId) => {
    try {
      setSelectedConversationId(convoId);
      const response = await fetch(`/api/textgen/conversations/${convoId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (!response.ok) throw new Error('Failed to load conversation');
      const messages = await response.json();
      
      setMessages(messages.map(m => ({
        id: m.id,
        text: m.text,
        sender: m.is_user ? 'user' : 'assistant',
        timestamp: m.created_at
      })));
      
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  };

  // Add a new function to handle starting a new conversation
  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setSelectedConversationId(null);
  }, []);

  // Add delete conversation handler
  const handleDeleteConversation = async (convoId) => {
    if (!window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`/api/textgen/conversations/${convoId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to delete conversation');
      }

      // Update conversations list
      setConversations(prev => prev.filter(conv => conv.id !== convoId));
      
      // If the deleted conversation was selected, clear the selection
      if (selectedConversationId === convoId) {
        setSelectedConversationId(null);
        setMessages([]);
      }

    } catch (error) {
      console.error('Error deleting conversation:', error);
      alert('Failed to delete conversation. Please try again.');
    }
  };

  useEffect(() => {
    console.log("Token in localStorage:", localStorage.getItem('token'));
  }, []);

  return (
    <main className={`${mainStyles} flex-1 flex relative`}>
      {/* Add sidebar toggle button */}
      <button 
        onClick={toggleSidebar}
        aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
        aria-expanded={isSidebarOpen}
        className={`
          fixed top-4 z-30 p-2 rounded-lg 
          bg-gray-800 hover:bg-gray-700 text-gray-300
          border border-gray-700 hover:border-gray-600
          ${isSidebarOpen ? 'left-[17rem]' : 'left-4'}
        `}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Wrap ConversationSidebar in conditional render */}
      {isSidebarOpen && (
        <ConversationSidebar
          conversations={conversations}
          selectedId={selectedConversationId}
          onSelect={handleSelectConversation}
          onLogout={logout}
          onNewConversation={handleNewConversation}
          onDelete={handleDeleteConversation}
        />
      )}

      <div className="flex-1 flex flex-col min-w-0">
        {/* Attach the ref to the messages container */}
        <div 
          ref={containerRef}
          className="bg-gray-950 w-full h-full p-4 md:px-8 overflow-y-auto"
        >
          <div className="max-w-4xl mx-auto flex flex-col gap-2">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </div>
        </div>

        <form onSubmit={handleSendMessage} className="bg-gray-800 p-4 border-t border-gray-700">
          <div className="max-w-5xl mx-auto bg-gray-900 rounded-lg border border-gray-700">
            <div className="flex-1 p-2">
              <textarea
                ref={textareaRef}
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                placeholder="Type your message..."
                className="w-full bg-transparent text-gray-100 resize-none outline-none placeholder-gray-500 rounded-lg transition-all"
                style={{ minHeight: '40px', maxHeight: '500px' }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isSending) {
                    handleSendMessage(e);
                  }
                }}
                aria-label="Type your message"
              />
            </div>
            
            <div className="bg-gray-900 border-t border-gray-900 w-full flex justify-between items-center p-2 rounded-b-lg">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-400">RAG:</label>
                <button
                  type="button"
                  onClick={() => setIsRagEnabled(!isRagEnabled)}
                  className={`px-2 py-1 text-xs rounded-md transition-colors ${
                    isRagEnabled 
                      ? 'bg-green-600 text-white hover:bg-green-700' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  aria-label="Toggle RAG"
                >
                  {isRagEnabled ? 'Enabled' : 'Disabled'}
                </button>
              </div>
              
              <button
                type="submit"
                disabled={!newMessage.trim() || isSending}
                className="bg-blue-600 text-white text-sm px-3 py-1.5 rounded-md border border-blue-500 hover:bg-blue-700 hover:border-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:border-blue-500"
                aria-label="Send message"
              >
                {isSending ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </main>
  );
} 