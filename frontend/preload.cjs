const { contextBridge, ipcRenderer } = require('electron')

// Expose only the specific operations the UI needs — not raw ipcRenderer, which
// would let any script in the page call any channel.
contextBridge.exposeInMainWorld('doclamar', {
  getBackendConfig: () => ipcRenderer.invoke('backend:config'),
  openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  getSettings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (settings) => ipcRenderer.invoke('settings:save', settings),
  openPath: (filePath) => ipcRenderer.invoke('shell:openPath', filePath),
  showInFolder: (filePath) => ipcRenderer.invoke('shell:showItemInFolder', filePath),
})
