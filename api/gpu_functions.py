# import torch
import humanize
import psutil
import GPUtil

def mem_report():
    '''
        Report memory usage.
    '''
    print("CPU memory usage: ", humanize.naturalsize(psutil.virtual_memory().available))
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        print("GPU {:d} memory free: {:.1f}/{:.1f}GB".format(i, gpu.memoryFree/1000, gpu.memoryTotal/1000))

