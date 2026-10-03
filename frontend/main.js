import { app, BrowserWindow, dialog, ipcMain, safeStorage, shell } from "electron";
import { spawn } from "child_process";
import crypto from "crypto";
import fs from "fs";
import net from "net";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isDev = !app.isPackaged;
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL || "http://localhost:5173";
const DOC_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];

// A fresh random token per launch: only this app's window can talk to the backend.
const backend = {
  url: null,
  port: null,
  token: crypto.randomBytes(32).toString("hex"),
  process: null,
  restarts: 0,
};
let quitting = false;
let mainWindow = null;

// ---------------------------------------------------------------- settings
// The LLM API key is encrypted with the OS keychain (DPAPI on Windows) and
// never shipped inside the installer.

const settingsPath = () => path.join(app.getPath("userData"), "settings.json");

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
  } catch {
    return {};
  }
}

function decryptKey(settings) {
  if (!settings.apiKey) return "";
  try {
    return safeStorage.decryptString(Buffer.from(settings.apiKey, "base64"));
  } catch {
    return "";
  }
}

function writeSettings({ provider, model, apiKey }) {
  if (!safeStorage.isEncryptionAvailable()) {
    throw new Error("Secure storage is not available, so the key can't be saved. It will work until you close the app.");
  }
  const data = {
    provider,
    model: model || "",
    apiKey: safeStorage.encryptString(apiKey).toString("base64"),
  };
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
  fs.writeFileSync(settingsPath(), JSON.stringify(data, null, 2), { mode: 0o600 });
}

function llmEnv() {
  const settings = readSettings();
  const key = decryptKey(settings);
  if (!settings.provider || !key) return {};
  const prefix = settings.provider.toUpperCase();
  return {
    LLM_PROVIDER: settings.provider,
    [`${prefix}_API_KEY`]: key,
    ...(settings.model ? { [`${prefix}_MODEL`]: settings.model } : {}),
  };
}

// ----------------------------------------------------------------- backend

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function backendCommand() {
  if (!isDev) {
    const exe = process.platform === "win32" ? "doclamar-backend.exe" : "doclamar-backend";
    const dir = path.join(process.resourcesPath, "doclamar-backend");
    return { cmd: path.join(dir, exe), args: [], cwd: dir };
  }
  const backendDir = path.join(__dirname, "..", "backend");
  const venvPython = process.platform === "win32"
    ? path.join(backendDir, ".venv", "Scripts", "python.exe")
    : path.join(backendDir, ".venv", "bin", "python");
  const python = process.env.DOCLAMAR_PYTHON || (fs.existsSync(venvPython) ? venvPython : "python");
  return { cmd: python, args: ["api.py"], cwd: backendDir };
}

async function startBackend() {
  if (process.env.DOCLAMAR_BACKEND_URL) {
    // Attach to a backend you started yourself (e.g. under a debugger).
    backend.url = process.env.DOCLAMAR_BACKEND_URL;
    backend.token = process.env.DOCLAMAR_TOKEN || "";
    return;
  }
  backend.port = backend.port || (await freePort());
  backend.url = `http://127.0.0.1:${backend.port}`;
  const { cmd, args, cwd } = backendCommand();
  console.log(`Starting backend: ${cmd} ${args.join(" ")} on port ${backend.port}`);

  const child = spawn(cmd, args, {
    cwd,
    windowsHide: true,
    stdio: ["pipe", "pipe", "pipe"], // backend exits when our end of stdin closes
    env: {
      ...process.env,
      ...llmEnv(),
      DOCLAMAR_TOKEN: backend.token,
      DOCLAMAR_PORT: String(backend.port),
      DOCLAMAR_EXIT_ON_STDIN_EOF: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONIOENCODING: "utf-8",
    },
  });
  backend.process = child;
  child.stdout.on("data", (d) => process.stdout.write(`[backend] ${d}`));
  child.stderr.on("data", (d) => process.stderr.write(`[backend] ${d}`));
  child.on("error", (err) => console.error("Could not start the backend:", err));
  child.on("exit", (code) => {
    backend.process = null;
    if (quitting) return;
    console.error(`Backend exited with code ${code}`);
    if (backend.restarts < 3) {
      backend.restarts += 1;
      setTimeout(startBackend, 1000);
    }
  });
}

async function backendPost(pathname, body) {
  const res = await fetch(`${backend.url}${pathname}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Doclamar-Token": backend.token },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Backend error ${res.status}`);
  return data;
}

// ------------------------------------------------------------------ window

function isAppUrl(url) {
  return isDev ? url.startsWith(DEV_SERVER_URL) : url.startsWith("file://");
}

async function loadDevServer(attempt = 0) {
  try {
    await mainWindow.loadURL(DEV_SERVER_URL);
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } catch {
    if (attempt < 40) setTimeout(() => loadDevServer(attempt + 1), 500); // Vite still starting
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: "DocLamar",
    backgroundColor: "#0a0a0a",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  // Links in answers open in the real browser; the app window never navigates away.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (isAppUrl(url)) return;
    event.preventDefault();
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
  });

  if (isDev) loadDevServer();
  else mainWindow.loadFile(path.join(__dirname, "dist", "index.html"));
}

// --------------------------------------------------------------------- IPC

function registerIpc() {
  ipcMain.handle("backend:config", () => ({ url: backend.url, token: backend.token }));

  ipcMain.handle("dialog:openDirectory", async () => {
    const result = await dialog.showOpenDialog(mainWindow, { properties: ["openDirectory"] });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dialog:openFile", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      filters: [{ name: "Documents", extensions: DOC_EXTENSIONS.map((e) => e.slice(1)) }],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("settings:get", () => {
    const settings = readSettings();
    return {
      provider: settings.provider || "groq",
      model: settings.model || "",
      hasKey: Boolean(decryptKey(settings)),
      secureStorage: safeStorage.isEncryptionAvailable(),
    };
  });

  ipcMain.handle("settings:save", async (_event, { provider, apiKey, model }) => {
    const saved = readSettings();
    const key = apiKey || (saved.provider === provider ? decryptKey(saved) : "");
    if (!key) throw new Error("Enter an API key.");
    // Validate with the provider before persisting anything.
    const llm = await backendPost("/config/llm", { provider, api_key: key, model: model || null, validate_key: true });
    writeSettings({ provider, model, apiKey: key });
    return llm;
  });

  ipcMain.handle("shell:openPath", async (_event, filePath) => {
    // Only open documents, never executables, whatever the renderer asks for.
    if (typeof filePath !== "string" || !path.isAbsolute(filePath)) return "Invalid path";
    if (!DOC_EXTENSIONS.includes(path.extname(filePath).toLowerCase())) return "Not a document";
    if (!fs.existsSync(filePath)) return "File not found";
    return shell.openPath(filePath);
  });

  ipcMain.handle("shell:showItemInFolder", (_event, filePath) => {
    if (typeof filePath === "string" && path.isAbsolute(filePath) && fs.existsSync(filePath)) {
      shell.showItemInFolder(filePath);
    }
  });
}

// --------------------------------------------------------------- lifecycle

if (!app.requestSingleInstanceLock()) {
  app.quit(); // a second copy would fight over the same index database
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    registerIpc();
    await startBackend();
    createWindow();
    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });

  app.on("before-quit", () => {
    quitting = true;
    if (backend.process) {
      backend.process.stdin.end();
      backend.process.kill();
    }
  });
}
