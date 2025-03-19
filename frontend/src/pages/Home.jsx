import React, { useState, useRef, useEffect, useCallback } from 'react';

// Move dummy data to constants at top
const DUMMY_CONVERSATIONS = [
  { id: 1, title: "Chat about React props" },
  { id: 2, title: "Project discussion" },
  { id: 3, title: "Code review notes" },
  { id: 4, title: "API integration talk" },
  { id: 5, title: "UX feedback session" },
];

const INITIAL_MESSAGES = [
  { id: 1, text: `Hi there! I'm your AI assistant. How can I help you today?`, sender: 'assistant' },
  { id: 2, text: `I'm trying to understand how to implement a responsive grid layout in Tailwind CSS. Can you explain the best approach?`, sender: 'user' },
  { id: 3, text: "Absolutely! Tailwind makes grid layouts straightforward. Here's a basic example:\n\n<div class=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">\n  <!-- Grid items -->\n</div>\n\nThis creates a responsive grid that shows 1 column on mobile, 2 on medium screens, and 3 on large screens.", sender: 'assistant' },
  { id: 4, text: "That makes sense! What about handling complex data fetching in React components? Should I use useEffect directly or a library like TanStack Query?", sender: 'user' },
  { id: 5, text: "For most applications, I recommend using TanStack Query (formerly React Query). It provides better caching, deduping, and error handling out of the box compared to raw useEffect implementations. It also simplifies loading states and pagination.", sender: 'assistant' },
  { id: 6, text: "Can you show me a basic example of using TanStack Query to fetch data from an API?", sender: 'user' },
  { id: 7, text: "Sure! Here's a basic example:\n\nimport { useQuery } from '@tanstack/react-query';\n\nfunction UserProfile({ userId }) {\n  const { data, isLoading, error } = useQuery({\n    queryKey: ['user', userId],\n    queryFn: () =>\n      fetch(`/api/users/${userId}`).then(res => res.json())\n  });\n\n  if (isLoading) return 'Loading...';\n  if (error) return 'Error: ' + error.message;\n\n  return (\n    <div>\n      <h2>{data.name}</h2>\n      <p>{data.email}</p>\n    </div>\n  );\n}", sender: 'assistant' },
  { id: 8, text: "Thanks! Now about state management - when would you recommend using Zustand over Redux Toolkit?", sender: 'user' },
  { id: 9, text: "Zustand is great for simpler apps where you want a lightweight solution. Redux Toolkit is better for complex state interactions with features like:\n- Built-in devtools\n- Middleware support\n- Larger ecosystem\n- Better for team collaboration\n- Advanced state persistence\n\nZustand's API is more minimal and might feel more natural if you're familiar with React hooks.", sender: 'assistant' },
  { id: 10, text: "How do I handle forms in React? Should I use a library like React Hook Form or build my own solution?", sender: 'user' },
  { id: 11, text: "Definitely use React Hook Form. It provides:\n- Efficient re-renders\n- Built-in validation\n- Easy integration with UI libraries\n- Better performance\n- Support for complex nested forms\n\nHere's a quick example:\n\nimport { useForm } from 'react-hook-form';\n\nfunction LoginForm() {\n  const { register, handleSubmit } = useForm();\n  \n  const onSubmit = data => console.log(data);\n\n  return (\n    <form onSubmit={handleSubmit(onSubmit)}>\n      <input {...register('email')} />\n      <input {...register('password')} type=\"password\" />\n      <button type=\"submit\">Login</button>\n    </form>\n  );\n}", sender: 'assistant' },
  { id: 12, text: "What's the best way to optimize React component renders? I'm seeing some performance issues in my app.", sender: 'user' },
  { id: 13, text: "Key optimization strategies:\n1. Use React.memo for pure components\n2. Implement useCallback for stable function references\n3. Use useMemo for expensive calculations\n4. Virtualize long lists (react-window)\n5. Profile with React DevTools\n6. Code splitting with React.lazy\n7. Avoid unnecessary context updates\n8. Keep component state local when possible\n\nWould you like me to elaborate on any of these?", sender: 'assistant' }
];

const ConversationSidebar = ({ conversations, selectedId, onSelect }) => (
  <div className="w-64 h-full bg-gray-800 border-r border-gray-700 flex flex-col">
    <div className="p-4 border-b border-gray-700">
      <h1 className="text-white text-xl font-bold">ChatApp</h1>
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
          {convo.title}
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
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [newMessage, setNewMessage] = useState('');
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef(null);
  const containerRef = useRef(null);

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

  // Update handleSendMessage to use API endpoint
  const handleSendMessage = async (e) => {
    e.preventDefault();
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || isSending) return;

    // Create user message
    const userMessage = {
      id: crypto.randomUUID(),
      text: trimmedMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    // Create temporary assistant loading message
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

      // Call textgen API endpoint
      const response = await fetch('/api/textgen/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(m => ({
            role: m.sender,
            content: m.text
          }))
        })
      });

      if (!response.ok) throw new Error('API request failed');

      const data = await response.json();

      // Replace loading message with actual response
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingMessage.id),
        {
          id: crypto.randomUUID(),
          text: data.content,
          sender: 'assistant',
          timestamp: new Date().toISOString()
        }
      ]);
    } catch (error) {
      console.error('Error:', error);
      // Add error message to UI
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
      // Add slight delay to ensure render completes
      setTimeout(() => {
        textareaRef.current?.focus();
        // Restore cursor position
        textareaRef.current?.setSelectionRange(
          textareaRef.current.value.length,
          textareaRef.current.value.length
        );
      }, 0);
    }
  };

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
          conversations={DUMMY_CONVERSATIONS}
          selectedId={selectedConversation}
          onSelect={setSelectedConversation}
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