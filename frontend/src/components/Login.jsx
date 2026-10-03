import React, { useState } from 'react';

const Login = ({ onLogin }) => {
  const [name, setName] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (name.trim()) onLogin(name);
  };

  return (
    <div className="fullscreen-view" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <form onSubmit={handleSubmit} className="glass-panel" style={{ padding: '40px', borderRadius: '16px', textAlign: 'center', width: '300px' }}>
        <h1 style={{ marginBottom: '10px' }}>DocLamar</h1>
        <p style={{ color: '#aaa', marginBottom: '30px' }}>Welcome back</p>
        <input 
          type="text" 
          placeholder="Enter your name..." 
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ width: '100%', padding: '12px', marginBottom: '20px', borderRadius: '8px', border: '1px solid #333', background: 'rgba(0,0,0,0.5)', color: 'white' }}
          autoFocus
        />
        <button type="submit" style={{ width: '100%', padding: '12px', background: '#ED7D31', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>
          Enter Workspace
        </button>
      </form>
    </div>
  );
};

export default Login;