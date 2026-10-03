import React from 'react';

const Home = ({ username, onEnterApp, isBackendReady }) => {

    
  return (
    <div className="fullscreen-view" style={{ padding: '50px', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
        <h2>Hello, {username}</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Status Indicator */}
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: isBackendReady ? '#4ade80' : '#facc15', boxShadow: `0 0 8px ${isBackendReady ? '#4ade80' : '#facc15'}` }}></div>
          <span style={{ color: '#aaa', fontSize: '0.9rem' }}>
            Engine Status: {isBackendReady ? 'Online' : 'Waking up...'}
          </span>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', flex: 1 }}>
        <div className="glass-panel" style={{ padding: '30px', borderRadius: '16px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', border: '1px solid rgba(255,255,255,0.1)' }}>
          <h3 style={{ marginBottom: '15px' }}>Ready to work?</h3>
          <p style={{ color: '#aaa', textAlign: 'center', marginBottom: '25px' }}>Start scanning directories or chatting with specific documents.</p>
          <button 
            onClick={onEnterApp}
            disabled={!isBackendReady}
            style={{ 
              padding: '12px 30px', 
              background: isBackendReady ? '#ED7D31' : '#555', 
              color: 'white', 
              border: 'none', 
              borderRadius: '8px', 
              cursor: isBackendReady ? 'pointer' : 'not-allowed',
              transition: 'background 0.3s'
            }}
          >
            {isBackendReady ? 'Open Chat Workspace' : 'Engine Loading...'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;