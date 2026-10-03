import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import QueryInput from './components/QueryInput'
import Login from './components/Login'   // <-- Import new component
import Home from './components/Home'     // <-- Import new component

export const parseDBDate = (dateString) => {
  if (!dateString) return new Date().toISOString();
  // Replaces the space with a 'T' and appends 'Z' for UTC time
  const safeDateString = dateString.replace(' ', 'T') + 'Z';
  return new Date(safeDateString).toISOString();
};

function App() {
  const savedName = localStorage.getItem('doclamar_username');
  const savedView = localStorage.getItem('doclamar_view');
  console.log("Memory Check on Refresh -> Name:", savedName, "| View:", savedView);
  
  // If there's a name, use the saved view (or default to 'home'). If no name, force 'login'.
  const initialView = savedName ? (savedView || 'home') : 'login';
  const [currentView, setCurrentView] = useState(initialView); // 'login', 'home', 'app'

  useEffect(() => {
    // Whenever currentView changes, save it to memory
    localStorage.setItem('doclamar_view', currentView);
  }, [currentView]);

  const [username, setUsername] = useState('');
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [directory, setDirectory] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [conversations, setConversations] = useState([])
  const [currentChatId, setCurrentChatId] = useState(null) // This will now store the SQLite session_id

  const [chatMode, setChatMode] = useState('directory'); 
  const [activeFile, setActiveFile] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/health");
        if (res.ok) setIsBackendReady(true);
      } catch (e) {
        // Backend not ready yet, silently try again in 2 seconds
        setTimeout(checkHealth, 2000);
      }
    };
    checkHealth();
  }, []);

  // --- 1. LOAD CHAT FROM SQLITE ---
  const handleLoadChat = async (sessionId) => {
  try {
    const response = await fetch(`http://127.0.0.1:8000/history/${sessionId}`);
    const data = await response.json();
    
    const dbMessages = data.messages;
    const pairedConversations = [];

    // Using a for-loop instead of .map() to safely group pairs
    for (let i = 0; i < dbMessages.length; i++) {
      const msg = dbMessages[i]; // <-- msg is explicitly defined right here

      if (msg.role === 'user') {
        pairedConversations.push({
          id: Date.now() + i,
          user: { 
            text: msg.content, 
            sender: 'user', 
            timestamp: parseDBDate(msg.created_at) 
          },
          assistant: null // Temporary placeholder until the AI response loops around
        });
      } else if (msg.role === 'ai' && pairedConversations.length > 0) {
        // Attach the AI response to the last created user pair
        pairedConversations[pairedConversations.length - 1].assistant = {
          text: msg.content,
          sender: 'assistant',
          timestamp: parseDBDate(msg.created_at), 
          citations: [] 
        };
      }
    }
    
    setConversations(pairedConversations);
    setCurrentChatId(sessionId);
    setChatMode('directory'); 
  } catch (err) {
    console.error("Failed to load session content:", err);
  }
};

  const handleChatWithFile = async (filePath, fileName) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/chat/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filePath })
      });
      
      const data = await response.json();
      
      setSessionId(data.session_id);
      setCurrentChatId(data.session_id); // Sync with sidebar
      setActiveFile(fileName);
      setChatMode('file');
      
      setConversations(prev => [...prev, {
        id: Date.now(),
        user: { text: `Focus on file: ${fileName}`, sender: 'system', timestamp: new Date().toISOString() },
        assistant: { 
          text: `**File Loaded:** I am now answering questions strictly based on the contents of **${fileName}**.`, 
          citations: null, 
          sender: 'assistant', 
          timestamp: new Date().toISOString() 
        }
      }]);
    } catch (error) {
      console.error("Failed to load file:", error);
    }
  };

  const handleReturnToDirectory = () => {
    setChatMode('directory');
    setActiveFile(null);
    setSessionId(null);
    setCurrentChatId(null);
    setConversations(prev => [...prev, {
      id: Date.now(),
      user: { text: `Return to directory search`, sender: 'system', timestamp: new Date().toISOString() },
      assistant: { text: `**Directory Mode:** I am now searching across all files in your selected folder again.`, citations: null, sender: 'assistant', timestamp: new Date().toISOString() }
    }]);
  };

  const handleDirectorySubmit = (path) => {
    setDirectory(path)
  }

  const handleNewChat = () => {
    setConversations([])
    setCurrentChatId(null)
    setSessionId(null)
    // Keep the directory so the user doesn't have to re-select it
  }

  const handleNewMessage = async (message) => {
    if (!directory) {
      alert('Please select a directory first')
      return
    }

    const tempId = Date.now();
    // 👇 NEW: Decide the Session ID BEFORE calling the backend
    const targetSessionId = currentChatId || `session_${tempId}`;

    const newConversationPair = {
      id: tempId,
      user: { id: tempId + 1, text: message, sender: 'user', timestamp: new Date().toISOString() },
      assistant: { text: "Thinking...", sender: 'assistant', timestamp: new Date().toISOString() }
    };
    
    setConversations(prev => [...prev, newConversationPair]);

    try {
      let endpoint = "http://127.0.0.1:8000/chat";
      let payload = { 
        query: message, 
        directory: directory,
        username: username,
        session_id: targetSessionId // <--- 👇 ADD IT TO THE PAYLOAD
      };

      if (chatMode === 'file' && sessionId) {
        endpoint = "http://127.0.0.1:8000/chat/message";
        payload = { message: message, session_id: sessionId, username: username };
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Backend failed");
      const data = await response.json();

      // 👇 Update this block to use your targetSessionId
      if (!currentChatId) {
          setCurrentChatId(targetSessionId); 
          setRefreshTrigger(prev => prev + 1);
      }

      setConversations(prev => prev.map(conv => {
        if (conv.id === tempId) {
          return {
            ...conv,
            assistant: {
              text: data.response,
              citations: data.citations,
              sender: 'assistant',
              timestamp: new Date().toISOString()
            }
          };
        }
        return conv;
      }));
    } catch (error) {
      console.error("Error communicating with AI backend:", error);
    }
  }

  const handleLogout = () => {
    // 1. Wipe the browser's memory
    localStorage.removeItem('doclamar_username');
    localStorage.removeItem('doclamar_view');
    
    // 2. Reset the React state
    setUsername('');
    setCurrentView('login');
  };

if (currentView === 'login') {
    return <Login onLogin={(name) => {
      // 👇 ADD THIS LINE 👇
      localStorage.setItem('doclamar_username', name); 
      
      setUsername(name);
      setCurrentView('home');
    }} />;
  }

  // And while you're here, add your logout function to the Home screen!
  if (currentView === 'home') {
    return <Home 
      username={username} 
      isBackendReady={isBackendReady} 
      onEnterApp={() => setCurrentView('app')} 
      onLogout={handleLogout} // <--- Don't forget this if you added the logout function!
    />;
  }
  return (
    <div className="app-container">
      <Sidebar 
        directory={directory} 
        onDirectorySubmit={handleDirectorySubmit}
        isProcessing={isProcessing}
        onLoadChat={handleLoadChat} // Now takes sessionId
        onNewChat={handleNewChat}
        currentChatId={currentChatId}
        chatMode={chatMode}
        activeFile={activeFile}
        onReturnToDirectory={handleReturnToDirectory}
        refreshTrigger={refreshTrigger}
        username={username} 
      />
      <main className="main-content">
        <ChatWindow 
          messages={conversations} 
          onChatWithFile={handleChatWithFile} 
        />
        <QueryInput onSubmit={handleNewMessage} isDisabled={!directory} />
      </main>
    </div>
  )
}

export default App