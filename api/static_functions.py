import matplotlib.pyplot as plt
import numpy as np
import sys

# the function to find index from the oversampled space for the sampled time point
def findIdx(value_s, vector):
    value_diff = np.abs( vector - value_s)
    resultA = np.where( value_diff == np.amin(value_diff) )
    if len(resultA[0])> 1:
        idx = int(resultA[0][0])
    else:
        idx = int( resultA[0])
    return idx
def createlabels(ticksLabelList): 
    ticksL = []
    for i in range(len(ticksLabelList)):
        if np.max(ticksLabelList)<10:
            ticksL.append("%.2f" % (ticksLabelList[i]))
        else:
            ticksL.append("%d" % (round(ticksLabelList[i])) )
    return ticksL
# a function to edit plots in the script
def compressPlot(inputArray,maxDispF, zeroThr, Ft,Fx,ft,fx,dispRange,xlabeltxt, ylabeltxt, titletxt, compress=False, vrange=None):
    # first: crop and resize image:
    idxt = [findIdx(dispRange[0], ft), findIdx(dispRange[1], ft)]
    idxx = [findIdx(dispRange[2], fx), findIdx(dispRange[3], fx)]
    # crop out the part outside the display range
    arr = np.array(inputArray[idxx[0]:idxx[1], idxt[0]:idxt[1]])
    vmin = 0 if vrange is None else vrange[0]
    vmax = arr.max()*maxDispF if vrange is None else vrange[1]
    if vmin == vmax:
        print("ERROR:", vmin, vmax, maxDispF, arr.max(), (arr == 0).sum()/arr.size)
    cmap = 'gray' if inputArray.ndim == 2 else None
    plt.imshow(arr, vmin = vmin, vmax = vmax, cmap=cmap)
    # now eidt the ticks and labels
    # first resize ft and fx
    width = (arr.shape)[1]
    height = (arr.shape)[0]
    tticksP = np.array( [1, width/2, width-1] )
    xticksP = np.array( [1, height/2, height-1] )
    tticksL = [dispRange[0], np.mean(dispRange[0:2]), dispRange[1]]
    xticksL = [dispRange[2], np.mean(dispRange[2:4]), dispRange[3]]

    ttickslabels = createlabels(tticksL)
    xtickslabels = createlabels(xticksL)
    plt.xlim( [0, width - 1] )
    plt.ylim( [0, height - 1] )
    lbFontSize = 28
    ttlFontSize = 36

    plt.xticks( tticksP, ttickslabels,fontsize = lbFontSize)
    plt.yticks( xticksP, xtickslabels,fontsize = lbFontSize)
    plt.gca().set_aspect("auto") #(idxt[1] - idxt[0])/(idxx[1] - idxx[0])
    plt.xlabel(xlabeltxt,fontsize = lbFontSize)
    plt.ylabel(ylabeltxt,fontsize = lbFontSize)
    plt.title(titletxt, fontsize = ttlFontSize)

    return
def calFlashT(frmRate,nFlash,holdInterval,RGBmode):
    if RGBmode == 'seq':
        flashT = 1/frmRate/nFlash*holdInterval/3
    else:
        flashT = 1/frmRate/nFlash*holdInterval
    return flashT
def updateHold(frmRate,nFlash,holdInterval,RGBmode,pxlResponseT):
    flashT = calFlashT(frmRate,nFlash,holdInterval,RGBmode)
    if flashT <= (2*pxlResponseT):
        print('cannot acheive this presentation rate with current pixel response time and hold interval')
        print('now increasing hold interval')
    while flashT <= (2*pxlResponseT) and (holdInterval<1):
        holdInterval += 0.1
        flashT = calFlashT(frmRate,nFlash,holdInterval,RGBmode)
    if holdInterval > 1: 
        holdInterval = 1
    print('current hold interval: %.2f'%holdInterval)
    return holdInterval
def updateP(frmRate,nFlash,holdInterval,RGBmode,pxlResponseT):
    flashT = calFlashT(frmRate,nFlash,holdInterval,RGBmode)
    if flashT <= 2*pxlResponseT: 
        if nFlash > 1: 
            print('recover the hold interval, now decreasing presentation rate')
            while flashT <= (2*pxlResponseT) and (nFlash > 1):
                nFlash -=1
                flashT = calFlashT(frmRate,nFlash,holdInterval,RGBmode)
            print('current number of flashes per frame: %d' %nFlash)    
    return nFlash

if __name__ == "__main__":
    args = sys.argv[1:]
    for i in args:
        try:
            eval(i+'()')
        except:
            print(i + " is not a defined function.")
