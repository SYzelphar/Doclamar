import { app, BrowserWindow, dialog, ipcMain } from "electron";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process"; // <-- Add this import
import fs from "fs"; // <-- ADD THIS IMPORT

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = !app.isPackaged;

let backendProcess = null;

function startBackend() {
  if (isDev) return;

  // 1. Detect if Windows to add .exe
  let executableName = "doclamar-backend";
  if (process.platform === "win32") {
    executableName += ".exe";
  }

  // 2. Resolve the path based on your package.json extraResources
  const backendPath = path.join(process.resourcesPath, "doclamar-backend", executableName);
  
  console.log("Attempting to start backend at:", backendPath);

  // 3. Ensure macOS allows the file to be executed (ignored on Windows)
  try {
    if (fs.existsSync(backendPath) && process.platform !== "win32") {
      fs.chmodSync(backendPath, '755');
    }
  } catch (err) {
    console.error("Failed to set permissions:", err);
  }

  // 4. Spawn the process
  backendProcess = spawn(backendPath, [], { 
    detached: false, 
    cwd: process.resourcesPath // Force Python to look here for the .env file!
  });

  backendProcess.stdout.on('data', (data) => console.log(`Backend: ${data}`));
  backendProcess.stderr.on('data', (data) => console.error(`Backend Error: ${data}`));
  
  // Catch spawn errors (e.g. if the .exe isn't where we expect it to be)
  backendProcess.on('error', (err) => {
    console.error("Failed to start backend process. Check if the path is correct in extraResources.", err);
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1920,
    height: 1080,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  if (isDev) {
    // Load Vite dev server
    win.loadURL("http://localhost:5173");
    win.webContents.openDevTools();
  } else {
    // Load built production files
    win.loadFile(path.join(__dirname, "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  // Handle file dialog
  ipcMain.handle('dialog:openDirectory', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory']
    });
    return result;
  });
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on('will-quit', () => {
  if (backendProcess) {
    backendProcess.kill();
  }
});
