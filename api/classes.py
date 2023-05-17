from parameter_dependent_functions import *
from parameter_dependent_functions import runWithErrorCatcher
import matplotlib.pyplot as plt
from base64 import b64encode
plt.ioff()
from threading import Thread, Timer
from queue import SimpleQueue
from waitress import serve
import os, datetime

class Logger(object):
    def __init__(self, stdout):
        folderPath = os.path.expanduser('~') + '/api/logs'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.fname = folderPath + '/' + datetime.datetime.now().isoformat() + '.txt'
        self.file = open(self.fname, 'a')
        self.stdout = stdout
    def write(self, inp):
        self.file.write(inp)
        self.stdout.write(inp)
        self.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()

class User():
    def __init__(self, allowedUsers):
        self.name = None
        self.addr = None
        self.timer = None
        self.timerTemplate = []
        self.allowedUsers = allowedUsers
    def setUser(self, name, addr):
        if self.name == None and name in self.allowedUsers:
            self.name = name
            self.addr = addr
            print('SET USERNAME TO: ' + name)
            print('SET USER IP TO: ' + addr)
            self.resetTimer()
            return True
    def getUser(self):
        return self.name
    def verifyAddress(self, addr):
        return self.addr == addr
    def clearUser(self):
        self.name = None
        self.addr = None
        self.resetTimer()
    def setTimer(self, minutes, function):
        if self.timer != None:
            self.timer.cancel()
        self.timer = Timer(minutes*60, function)
        self.timerTemplate = [minutes*60, function]
        self.timer.start()
    def resetTimer(self):
        try:
            self.timer.cancel()
            self.timer = Timer(*self.timerTemplate)
            self.timer.start()
        except Exception as e:
            print("TIMER RESET EXCEPTION: " + e)


class ProgressTracker():
    def __init__(self, message, compareInd=None, stereoCompareInd=None):
        self.queue = SimpleQueue()
        self.message = message
        self.progress = 0
        self.compareInd = compareInd
        self.pauseCompare = False
        self.stereoCompareInd = stereoCompareInd
        self.stereoPauseCompare = False
        self.device = None
    def setProgress(self, message, progress):
        self.message = message
        self.progress = progress
        self.queue.put((message, progress))
    def reset(self, message=""):
        self.message = message
        self.progress = 0
        self.queue = SimpleQueue()
    def resetCompare(self):
        self.compareInd = None
        self.pauseCompare = False
        self.stereoCompareInd = None
        self.stereoPauseCompare = False
    def getProgress(self, dump=True):
        if self.queue.empty():
            return 'nop', 'nop'
        else:
            if dump:
                out = []
                while not self.queue.empty():
                    out.append(self.queue.get())
                return out
            return self.queue.get()
    def isActive(self):
        return not (self.message == "" and self.progress == 0)


class ToolboxRunner():
    def __init__(self):
        self.params = {}
        self.tracker = ProgressTracker("")
        self.thread = None
        self.key_mode = False
    def updateParams(self, params):
        self.params = params
    def run(self, stereo=False):
        print("\n\nActual Parameters", str(self.params), "\n\n")
        self.tracker.reset("Initialized Run: " + str(self.params))
        self.thread = Thread(target=lambda: runWithErrorCatcher(self.params, tracker=self.tracker, stereo=stereo))
        self.thread.start()
    def runSync(self, stereo=False):
        print("\n\nActual Parameters", str(self.params), "\n\n")
        self.tracker.reset("Initialized Run: " + str(self.params))
        return runWithErrorCatcher(self.params, tracker=self.tracker, stereo=stereo)
    def getProgress(self):
        prog = self.tracker.getProgress()
        return prog