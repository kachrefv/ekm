const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
        title: "EKM Desktop"
    });

    if (process.env.VITE_DEV_SERVER_URL) {
        mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }
}

function startPythonBackend() {
    const isDev = process.env.VITE_DEV_SERVER_URL;
    let scriptPath;
    let pythonExe;

    if (isDev) {
        // In development, run from the virtual environment
        pythonExe = path.join(__dirname, '../../../../.venv/Scripts/python.exe');
        scriptPath = path.join(__dirname, '../../server.py');
        pythonProcess = spawn(pythonExe, [scriptPath], {
            cwd: path.join(__dirname, '../../../../'),
            env: { ...process.env, PYTHONPATH: '.' }
        });
    } else {
        // In production, run the bundled executable
        scriptPath = path.join(process.resourcesPath, 'server.exe');
        pythonProcess = spawn(scriptPath);
    }

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
}

app.whenReady().then(() => {
    startPythonBackend();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        if (pythonProcess) pythonProcess.kill();
        app.quit();
    }
});

app.on('will-quit', () => {
    if (pythonProcess) pythonProcess.kill();
});
