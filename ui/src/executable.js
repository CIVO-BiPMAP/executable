const isDev = require('electron-is-dev');
const { spawn } = require('child_process')
var treeKill = require('tree-kill');
const path = require('path');
const fs = require('fs');
const urlExists = require('url-exists');

function localPath() {
  return path.join(__dirname, '..', ...arguments);
}

function launchCommand(command, nickname) {
  const tmp = spawn(command, {shell: true})
  console.log(nickname, 'pid', tmp.pid)
  tmp.on('exit', function (code, signal) {
    console.log(nickname, 'process exited with', 
                `code ${code} and signal ${signal}`);
    if (code === 0) {
      console.log(nickname, "succeeded.")
    }
  });
  tmp.stdout.on('data', (data) => {
    console.log(`${nickname} stdout:\n${data}`);
  });
  
  tmp.stderr.on('data', (data) => {
    console.error(`${nickname} stderr:\n${data}`);
  });
  return tmp;
}

function launchPromiseCommand(command, nickname) {
  return new Promise((resolve, reject) => {
    const tmp = launchCommand(command, nickname)
    tmp.on('exit', function (code, signal) {
      console.log(nickname, 'process exited with', 
                  `code ${code} and signal ${signal}`);
      if (code === 0) {
        console.log(nickname, "succeeded.")
        resolve();
      } else {
        reject();
      }
    });
  });
}

function py(command) {
  return `(python ${command} || py ${command} || python3 ${command})`
}

function appWindow(win){
  //load the html content from the backend. if no content, then reload every second until content is available
  if (isDev) {
    win.loadURL('http://localhost:3333');
    let interval = setInterval(() => {
      urlExists('http://localhost:3333', (_, exists) => (exists ? win.reload() || clearInterval(interval) : null));
    }  , 1000);
    win.webContents.openDevTools();
  } else {
    win.loadFile('build/index.html');
  }
  win.removeMenu();
  win.maximize();
  win.show();
}

function createVenvCommand() {
  // win.loadURL('file://'+localPath('src', 'loading.html'));
  let command0 = py(`-m venv ${localPath('..', 'toolbox')}`)
  let venv = localPath('..', 'toolbox', 'Scripts', 'activate')
  let command1 = py(`-m pip install --upgrade pip setuptools wheel`)
  let command2 = py(`-m pip install -r ${localPath('..', 'api', 'requirements.txt')}`)
  // Parse version of CUDA
  let command3 = `nvidia-smi ^| findstr /r "CUDA Version:"`
  // Assign version to variable
  command3 = `(for /f "usebackq tokens=*" %i in (\`${command3}\`) do (set cuda_v=%i))`
  // install version specific cupy
  command3 = `(cmd.exe /v /c "${command3} && (if [!cuda_v!] neq [] ${py("-m pip install cupy-cuda!cuda_v:~-10,2!!cuda_v:~-7,1!")})")`
  return `(${command0} && ${venv} && ${command1} && ${command2} && ${command3}) > ${localPath('installog.txt')}`
}

const pypath = localPath('..', 'api', 'api.py') + (isDev ? " -d" : "")
let venv = localPath('..', 'toolbox', 'Scripts', 'activate')
let python_command = `(${venv} && ${py(pypath)}) > ${localPath('pylog.txt')}`
if (!fs.existsSync(venv)) {
  python_command = `((${createVenvCommand()}) && (${python_command}))`;
}

const python = launchCommand(python_command, "python");
const react = isDev ? launchCommand(`npm --prefix=${localPath()} run react-dev > ${localPath('reactlog.txt')}`, "react") : null

const { app, BrowserWindow } = require('electron')
app.allowRendererProcessReuse = false;
const electron = require('electron');
const dialog = electron.dialog;

// Disable/override error dialogs
dialog.showErrorBox = function(title, content) {
    console.log(`${title}\n${content}`);
};

function createWindow () {
  try {
    const win = new BrowserWindow({
      title:"BiPMAP",
      icon: __dirname + '/assets/ucb_logo.ico',
      webPreferences: {
        nodeIntegration: true
      },
      show: false
    })
    appWindow(win);
  } catch (err) {
    console.log("Error creating window: ", err)
    app.quit();
  }
}

app.whenReady().then(createWindow)

async function killAllTasks() {
  let killTask = (pid, nickname, force=false) => {
    let killcommand = `taskkill /pid ${pid} /T ${force ? '/f' : ''} || echo "failed to kill ${nickname}"`
    killcommand = `(${killcommand}) && echo "Succeeded in killing ${nickname}"`
    return launchPromiseCommand(killcommand, 'kill ' + nickname)
  }
  try {react.stdin.write("q\n")} catch (err) {}
  try {python.stdin.write("q\n")} catch (err) {}
  let react_killer = killTask(react.pid, 'React').catch((err) => {
    console.log("React kill error (ignore if using executable): ", err)
    return killTask(react.pid, 'React', true).catch((err) => {
      console.log("React kill error despite forcing (ignore if using executable): ", err)
    })
  })
  let python_killer = killTask(python.pid, 'Python').catch((err) => {
    console.log("Python kill error", err)
    return killTask(python.pid, 'Python', true).catch((err) => {
      console.log("Python kill error despite forcing", err)
    })
  })
  return Promise.allSettled([react_killer, python_killer])
}

function treeKillPromise(pid, signal='SIGKILL') {
  return new Promise((resolve, reject) => {
    treeKill(pid, signal, (err) => {
      if (`${pid}: ${err}`) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}
async function killAll() {
  //create array with promises to kill in this case react and python
  let tasks_to_kill = [react, python].filter((task) => task != null)
  let kill_promises = tasks_to_kill.map(
    (task) => treeKillPromise(task.pid, 'SIGKILL').catch(
      (err) => console.log("Task kill error: ", err)
    )
  )
  return Promise.allSettled(kill_promises).then(() => app.quit())
}

app.on('window-all-closed', () => {
  try {
    killAll()
  } catch (err) {
    console.log("Error on window close: ", err)
  } finally {
    app.quit();
  }
  return 0;
})

// app.on('quit', () => {
//   try {
//     killAll().then(() => app.exit());
//   } catch (err) {
//     console.log("Error on quit: ", err)
//     app.exit();
//   }
//   return 0;
// })

// // Open a new window if none are open (macOS)
// app.on('activate', () => {
//   if (BrowserWindow.getAllWindows().length === 0) {
//     createWindow()
//   }
// })