import { useRef, useEffect } from 'react'

const ChatWindow = ({ messages }) => {
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.map((conversation) => (
          <div key={conversation.id} className="conversation-pair">
            {/* User Message */}
            <div className="message user-message">
              <div className="message-content">
                <div className="message-avatar">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                  </svg>
                </div>
                <div className="message-text">
                  <span className="message-label">You</span>
                  {conversation.user.text}
                </div>
              </div>
              <div className="message-timestamp">
                {new Date(conversation.user.timestamp).toLocaleTimeString()}
              </div>
            </div>

            {/* Assistant Message */}
            <div className="message assistant-message">
              <div className="message-content">
                <div className="message-avatar">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M12 16v-4M12 8h.01"></path>
                  </svg>
                </div>
                <div className="message-text">
                  <span className="message-label">Assistant</span>
                  {conversation.assistant.text}
                  {conversation.assistant.citations && (
                    <div className="citations-container">
                      <h4>Citations:</h4>
                      <div className="citations-list">
                        {conversation.assistant.citations.map((citation, index) => (
                          <div key={index} className="citation-item">
                            <div className="citation-file">
                              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="16" y1="13" x2="8" y2="13"></line>
                                <line x1="16" y1="17" x2="8" y2="17"></line>
                                <polyline points="10 9 9 9 8 9"></polyline>
                              </svg>
                              <code>{citation.file}</code>
                            </div>
                            <div className="citation-snippet">{citation.snippet}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <div className="message-timestamp">
                {new Date(conversation.assistant.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}

export default ChatWindow
