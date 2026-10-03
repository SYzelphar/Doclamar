import { useState, useEffect } from 'react'

const Sidebar = ({ directory, onDirectorySubmit, isProcessing, savedChats, onLoadChat, onNewChat, currentChatId }) => {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [directoryInput, setDirectoryInput] = useState('')
  const [fullPath, setFullPath] = useState(null)

  // Update local input when directory prop changes
  const isAbsolutePath = (p) => {
    if (!p || typeof p !== 'string') return false
    // Windows absolute path e.g. C:\... or Unix-like /home/...
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

  const handleDeleteChat = (chatId) => {
    try {
      const existingSavedChats = JSON.parse(localStorage.getItem('savedChats') || '[]')
      const updatedChats = existingSavedChats.filter(chat => chat.id !== chatId)
      localStorage.setItem('savedChats', JSON.stringify(updatedChats))
      window.location.reload()
    } catch (error) {
      console.error('Error deleting chat:', error)
    }
  }

  const handleFileExplorer = async () => {
    try {
      // Try Electron first
      if (window.electron && window.electron.ipcRenderer) {
        const result = await window.electron.ipcRenderer.invoke('dialog:openDirectory')
        if (result.filePaths && result.filePaths.length > 0) {
          const selectedPath = result.filePaths[0]
          setDirectoryInput(selectedPath)
          setFullPath(selectedPath)
          onDirectorySubmit(selectedPath)
        }
      } 
      // Try File System Access API (modern browsers)
      else if (window.showDirectoryPicker) {
        try {
          const dirHandle = await window.showDirectoryPicker()
          const path = dirHandle.name
          // Browser API does NOT expose the absolute filesystem path for security reasons.
          // We store only the folder name and indicate that full path is unavailable.
          setDirectoryInput(path)
          setFullPath(null)
          onDirectorySubmit(path)
        } catch (error) {
          if (error.name !== 'AbortError') {
            console.error('Error accessing directory:', error)
          }
        }
      } 
      // Fallback: show message
      else {
        alert('File explorer not available in your browser. Please enter the directory path manually (e.g., C:\\Users\\YourName\\Documents or /home/user/documents)')
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
            <button
              type="button"
              className="send-button directory-browse-btn"
              onClick={handleFileExplorer}
              title="Browse for directory"
              aria-label="Browse for directory"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" role="img" aria-hidden="true">
                <title>File</title>
                <path fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" points="14 2 14 8 20 8"></polyline>
              </svg>
            </button>
          </div>
          {directory && (
            <div className="current-directory">
              <span>Current Directory:</span>
              {fullPath ? (
                <code title={fullPath}>{fullPath}</code>
              ) : (
                <div>
                  <code title={directory}>{directory}</code>
                  <div className="path-hint">(Full absolute path unavailable in browser mode. Paste full path into the field and click "Set", or run the app in Electron for absolute paths.)</div>
                </div>
              )}
            </div>
          )}
          {!directory && (
            <div className="current-directory-empty">
              <span className="hint">Select a directory to get started</span>
            </div>
          )}
        </div>
        <button 
          className="collapse-btn"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d={isCollapsed ? "M9 18l6-6-6-6" : "M15 18l-6-6 6-6"} />
          </svg>
        </button>
      </div>

      <div className="chat-history">
        <div className="chat-history-header">
          <h3>History</h3>
          <button 
            className="new-chat-btn"
            onClick={onNewChat}
            title="Start new chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New Chat
          </button>
        </div>
        
        {savedChats.length === 0 ? (
          <p className="empty-message">No chats yet. Start a conversation!</p>
        ) : (
          savedChats.map((chat) => (
            <div 
              key={chat.id}
              className={`chat-item ${currentChatId === chat.id ? 'active' : ''}`}
            >
              <div 
                className="chat-item-content"
                onClick={() => onLoadChat(chat)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
                <div className="chat-item-text">
                  <div className="chat-item-title">{chat.title}</div>
                  <div className="chat-item-meta">
                    {chat.conversations.length} messages • {new Date(chat.timestamp).toLocaleDateString()}
                  </div>
                </div>
              </div>
              <button 
                className="delete-chat-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  if (confirm('Are you sure you want to delete this chat?')) {
                    handleDeleteChat(chat.id)
                  }
                }}
                title="Delete chat"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  <line x1="10" y1="11" x2="10" y2="17"></line>
                  <line x1="14" y1="11" x2="14" y2="17"></line>
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
