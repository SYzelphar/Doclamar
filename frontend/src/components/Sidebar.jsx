import { useState, useEffect } from 'react'

const parseDBDate = (dateString) => {
  if (!dateString) return new Date();
  return new Date(dateString.replace(' ', 'T') + 'Z');
};

const Sidebar = ({ directory, onDirectorySubmit, isProcessing, onLoadChat, onNewChat, currentChatId, chatMode, activeFile, onReturnToDirectory,refreshTrigger,username }) => {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [directoryInput, setDirectoryInput] = useState('')
  const [fullPath, setFullPath] = useState(null)
  const [dbSessions, setDbSessions] = useState([]) // New state for SQLite sessions

  // --- FETCH HISTORY FROM SQLITE ---
  const fetchHistory = async (retries = 3) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/history?username=${username}`);
      if (response.ok) {
        const data = await response.json();
        setDbSessions(data.sessions);
      }
    } catch (error) {
      console.error('Error fetching SQLite history:', error);
      // If the backend is still waking up, try again in 1.5 seconds!
      if (retries > 0) {
        setTimeout(() => fetchHistory(retries - 1), 1500);
      }
    }
  }

  // Reload history when app starts or when a new chat becomes active
  useEffect(() => {
    fetchHistory()
  }, [currentChatId,refreshTrigger])

  const isAbsolutePath = (p) => {
    if (!p || typeof p !== 'string') return false
    return /^[a-zA-Z]:\\/.test(p) || p.startsWith('/')
  }

  useEffect(() => {
    if (directory) {
      setDirectoryInput(directory)
      setFullPath(isAbsolutePath(directory) ? directory : null)
    } else {
      setDirectoryInput('')
      setFullPath(null)
    }
  }, [directory])

  // --- DELETE FROM SQLITE ---
  const handleDeleteChat = async (sessionId) => {
    try {
      // Note: We need to add this endpoint to api.py next!
      const response = await fetch(`http://127.0.0.1:8000/history/${sessionId}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        fetchHistory() // Refresh the list
        if (currentChatId === sessionId) onNewChat() // Reset if we deleted active chat
      }
    } catch (error) {
      console.error('Error deleting chat from DB:', error)
    }
  }

  const handleFileExplorer = async () => {
    try {
      if (window.electron && window.electron.ipcRenderer) {
        const result = await window.electron.ipcRenderer.invoke('dialog:openDirectory')
        if (result.filePaths && result.filePaths.length > 0) {
          const selectedPath = result.filePaths[0]
          setDirectoryInput(selectedPath)
          setFullPath(selectedPath)
          onDirectorySubmit(selectedPath)
        }
      } 
      else if (window.showDirectoryPicker) {
        try {
          const dirHandle = await window.showDirectoryPicker()
          const path = dirHandle.name
          setDirectoryInput(path)
          setFullPath(null)
          onDirectorySubmit(path)
        } catch (error) {
          if (error.name !== 'AbortError') console.error('Error accessing directory:', error)
        }
      } 
    } catch (error) {
      console.error('Error opening file explorer:', error)
    }
  }

  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="directory-section">
          <h3>File Directory</h3>
          <div className="directory-form">
            <div className="directory-input-wrapper">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
              </svg>
              <input
                type="text"
                value={directoryInput}
                onChange={(e) => setDirectoryInput(e.target.value)}
                placeholder="Enter directory path..."
                className="directory-input"
              />
            </div>
            <button type="button" className="send-button directory-browse-btn" onClick={handleFileExplorer}>
               <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                <path fill="none" stroke="white" strokeWidth="1.5" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline fill="none" stroke="white" strokeWidth="1.5" points="14 2 14 8 20 8"></polyline>
              </svg>
            </button>
          </div>
          {/* ... Current Directory Display Code (unchanged) ... */}
        </div>
      </div>

      <div className="chat-history">
        <div className="chat-history-header">
          <h3>History</h3>
          <button className="new-chat-btn" onClick={onNewChat}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New Chat
          </button>
        </div>
        
        {dbSessions.length === 0 ? (
          <p className="empty-message">No chats yet. Start a conversation!</p>
        ) : (
          dbSessions.map((session) => (
            <div key={session.id} className={`chat-item ${currentChatId === session.id ? 'active' : ''}`}>
              <div className="chat-item-content" onClick={() => onLoadChat(session.id)}>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
                <div className="chat-item-text">
                  <div className="chat-item-title">{session.title}</div>
                  <div className="chat-item-meta">
  {parseDBDate(session.created_at).toLocaleDateString()}
</div>
                </div>
              </div>
              <button className="delete-chat-btn" onClick={(e) => {
                  e.stopPropagation()
                  if (confirm('Delete this chat history?')) handleDeleteChat(session.id)
                }}>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  )
}

export default Sidebar