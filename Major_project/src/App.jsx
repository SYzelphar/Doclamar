import { useState } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import QueryInput from './components/QueryInput'

function App() {
  const [directory, setDirectory] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [conversations, setConversations] = useState([])
  const [savedChats, setSavedChats] = useState([])
  const [currentChatId, setCurrentChatId] = useState(null)

  const handleDirectorySubmit = (path) => {
    setIsProcessing(true)
    // Here you would integrate with your ML backend to process the directory
    setDirectory(path)
    setIsProcessing(false)
  }

  const autoSaveChat = (conversationsToSave) => {
    try {
      const existingSavedChats = JSON.parse(localStorage.getItem('savedChats') || '[]')
      
      if (currentChatId) {
        // Update existing chat
        const chatIndex = existingSavedChats.findIndex(chat => chat.id === currentChatId)
        if (chatIndex !== -1) {
          existingSavedChats[chatIndex].conversations = conversationsToSave
          existingSavedChats[chatIndex].timestamp = new Date().toISOString()
        }
      } else {
        // Create new chat
        const newChatId = Date.now()
        setCurrentChatId(newChatId)
        
        const chatData = {
          id: newChatId,
          title: `Chat - ${new Date().toLocaleString()}`,
          directory: directory,
          conversations: conversationsToSave,
          timestamp: new Date().toISOString()
        }
        
        existingSavedChats.push(chatData)
      }
      
      localStorage.setItem('savedChats', JSON.stringify(existingSavedChats))
      setSavedChats(existingSavedChats)
    } catch (error) {
      console.error('Error auto-saving chat:', error)
    }
  }

  const handleNewMessage = (message) => {
    if (!directory) {
      alert('Please select a directory first')
      return
    }

    // Simulating response - replace with actual ML processing
    const response = {
      id: Date.now() + 1,
      text: "This is a sample response. Your ML model will provide the actual response based on the files in the directory.",
      sender: 'assistant',
      timestamp: new Date().toISOString(),
      citations: [
        { file: 'example/path/file1.txt', snippet: 'Relevant text from file 1' },
        { file: 'example/path/file2.txt', snippet: 'Relevant text from file 2' }
      ]
    }

    // Create conversation pair (user + assistant together)
    const conversationPair = {
      id: Date.now(),
      user: {
        id: Date.now(),
        text: message,
        sender: 'user',
        timestamp: new Date().toISOString()
      },
      assistant: response
    }

    const updatedConversations = [...conversations, conversationPair]
    setConversations(updatedConversations)
    
    // Auto-save to localStorage
    autoSaveChat(updatedConversations)
  }

  const handleLoadChat = (chat) => {
    setConversations(chat.conversations)
    setDirectory(chat.directory)
    setCurrentChatId(chat.id)
  }

  const handleNewChat = () => {
    setConversations([])
    setCurrentChatId(null)
    setDirectory('')
  }

  // Load saved chats on component mount
  const [isInitialized, setIsInitialized] = useState(false)
  if (!isInitialized) {
    try {
      const existingSavedChats = JSON.parse(localStorage.getItem('savedChats') || '[]')
      setSavedChats(existingSavedChats)
      setIsInitialized(true)
    } catch (error) {
      console.error('Error loading saved chats:', error)
      setIsInitialized(true)
    }
  }

  return (
    <div className="app-container">
      <Sidebar 
        directory={directory} 
        onDirectorySubmit={handleDirectorySubmit}
        isProcessing={isProcessing}
        savedChats={savedChats}
        onLoadChat={handleLoadChat}
        onNewChat={handleNewChat}
        currentChatId={currentChatId}
      />
      <main className="main-content">
        <ChatWindow messages={conversations} />
        <QueryInput onSubmit={handleNewMessage} isDisabled={!directory} />
      </main>
    </div>
  )
}

export default App
