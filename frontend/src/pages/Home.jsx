import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import authStore from '../stores/authStore';

const ConversationSidebar = ({ conversations, selectedId, onSelect, onLogout, onNewConversation }) => (
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
        <button
          key={convo.id}
          onClick={() => onSelect(convo.id)}
          className={`w-full p-2 mb-1 text-sm rounded cursor-pointer text-left transition-colors
            ${selectedId === convo.id 
              ? 'bg-gray-700 text-white' 
              : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`}
          aria-current={selectedId === convo.id}
        >
          <div className="flex justify-between items-center">
            <span>{convo.title}</span>
            <span className="text-xs text-gray-400">
              {convo.message_count} messages
            </span>
          </div>
        </button>
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

  // Update handleSendMessage to use conversation ID
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

      // Update API call to include conversation ID
      const response = await fetch('/api/textgen/generate', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}` 
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: trimmedMessage }],
          conversation_id: selectedConversationId
        })
      });

      if (!response.ok) throw new Error('API request failed');
      const data = await response.json();

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
          text: `Error: ${error.message}`,
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
            
            <div className="bg-gray-900 border-t border-gray-900 w-full flex justify-end p-2 rounded-b-lg">
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