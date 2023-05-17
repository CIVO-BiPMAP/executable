import traceback
import time
import math
import gc
from skimage import io
from static_functions import findIdx, compressPlot, updateHold, calFlashT, updateP
from matplotlib import gridspec
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib
matplotlib.use('Agg')
mplstyle.use('fast')
def identity(x=False):
    '''
        Trivial function.
    '''
    return x

try:
    import numpy as np
    from gpu_functions import mem_report
    import cupy as cp
    
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    def mem_clear():
        '''
        clear GPU memory.
        '''
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        gc.collect()
        
    NP2CP = cp.asarray
    CP2NP = cp.asnumpy
    print("GPU ACCESSIBLE", flush=True)
    IS_GPU = True
except Exception as e:
    IS_GPU = False
    import numpy as cp
    NP2CP = identity
    CP2NP = identity
    mem_clear = gc.collect
    print("GPU Not accessible. Falling back on CPU.", flush=True)
    print(e, flush=True)

mem_clear()
mem_report()    
FFTSHIFT = cp.fft.fftshift
FFT2 = cp.fft.fft2
IFFT2 = cp.fft.ifft2

def findIndex(value_s, vector, force_np=False):
    if force_np:
        abs_f = np.abs
    else:
        abs_f = cp.abs
    idx = abs_f(vector-value_s).argmin()
    return idx

def runWithErrorCatcher(dict, tracker, stereo=False, catch=False):
    init_time = time.time()
    def full_run():
        def track(message, percent):
            print(message, flush=True)
            if tracker != None:
                tracker.setProgress(message, percent)
            return None
        tracker.reset()

        def run(**args):
            if stereo:
                return stereo_parameters(**dict, track=track, tracker=tracker, **args)
            else:
                return parameters(**dict, track=track, tracker=tracker, **args)
        global FFTSHIFT, FFT2, IFFT2, NP2CP, CP2NP
        if IS_GPU and (tracker.device or 0) > 0:
            FFTSHIFT = cp.fft.fftshift
            FFT2 = cp.fft.fft2
            IFFT2 = cp.fft.ifft2
            NP2CP = cp.asarray
            CP2NP = cp.asnumpy
            with cp.cuda.Device(tracker.device - 1):
                track(f'GPU Compute: {cp.cuda.Device().compute_capability}', 0)
                run(gpu=True)
                return time.time() - init_time
        else:
            FFTSHIFT = np.fft.fftshift
            FFT2 = np.fft.fft2
            IFFT2 = np.fft.ifft2
            NP2CP = identity
            CP2NP = identity
            track('CPU Enabled', 0)
            run(gpu=False)
            return time.time() - init_time
    
    if catch:
        try:
            full_run()
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            if tracker != None:
                tracker.setProgress(
                    str(exc)+"\n(Possibly not enough memory in cpu/gpu)", -1)
    else:
        full_run()
def parameters(
        savePath="run_output.png",
        nFlash=1,
        pxlResponseT=0,
        frmRate=60,
        holdInterval=1,
        recordingLength=0.5,
        dpi=300,
        fillF=1,
        antialiasingF=0,
        vx=10,
        objSize=0.05,
        contrast=1,
        contrast_R=1,
        contrast_G=1,
        contrast_B=1,
        luminance=100,
        RGBmode='bw',
        ve=True,
        viewing_D=50,
        spatialOffset=False,
        track=print,
        tracker=None,
        gpu=False,
        **kwargs):
    
    print("Ignored kwargs:", kwargs)

    if gpu:
        import cupy as cp
        print("USING GPU")
    else:
        import numpy as cp
        print("USING CPU")

    init_time = time.time()
    checkpoint = init_time

    def trackTime(message, percent):
        nonlocal checkpoint
        now = time.time()
        track(message+' (%.1fs)' % (now-checkpoint), percent)
        checkpoint = time.time()
    FOVEA_SIZE = 5 #deg
    precision = 'double'
    def asdeg(cm_value, force_numpy=False): # convert cm to degree
        if force_numpy:
            return 2*np.arctan(cm_value / viewing_D / 2)/math.pi*180
        return 2*cp.arctan(cm_value / viewing_D / 2)/math.pi*180
    def ascm(deg_value, force_numpy=False): # convert degree to cm
        if force_numpy:
            return viewing_D * np.tan(deg_value/180*math.pi/2) * 2
        return viewing_D * cp.tan(deg_value/180*math.pi/2) * 2
    
    # disp_factor = None# (0.4, 0.17) #(0.6, 0.3) # crop factor, s.t. 0 means no crop, 0.5 means crop 50% of the image. 1-FOVEA_SIZE/visualRange
    disp_crop_bounds = (0.1, ascm(2, True)) #in s, deg
    iprecision = 'int32'
    ve = vx if ve else 0
    v = vx - ve
    contrast = [
        contrast_R,
        contrast_G,
        contrast_B
        ] if RGBmode != 'bw' else [contrast]*3
    contrast = np.array(contrast, dtype=precision)
    antialiasing = antialiasingF != 0
    plt.ioff()
    pxlSize = 2.54/dpi # pixel size, in cm
    strUpdate = '(holdInterval = %.2f)' % holdInterval
    x_oversample_factor = 6
    t_interval = 0.0001

    pxlResponseT = pxlResponseT/1000  # convert ms to s
    holdInterval_init = holdInterval
    flashT = calFlashT(frmRate, nFlash, holdInterval, RGBmode)
    holdInterval = updateHold(
        frmRate, nFlash, holdInterval, RGBmode, pxlResponseT)
    flashT = calFlashT(frmRate, nFlash, holdInterval, RGBmode)
    if flashT <= (2*pxlResponseT):
        holdInterval = holdInterval_init
        track(
            'Cannot acheive the current presentation rate, hold Interval value restored.', 0)
        nFlash = updateP(frmRate, nFlash, holdInterval, RGBmode, pxlResponseT)
        flashT = calFlashT(frmRate, nFlash, holdInterval, RGBmode)
        if flashT <= (2*pxlResponseT):
            track(
                'Cannot proceed with current frame rate, try increasing hold interval again.', 0)
            holdInterval = updateHold(
                frmRate, nFlash, holdInterval, RGBmode, pxlResponseT)
    if flashT <= (2*pxlResponseT):
        print('cannot proceed with current frame rate and pixel response time, please reduce frame rate', -1)
        return None
    track('Final presentation rate: %dX; Final hold interval: %.2f.' %
          (nFlash, holdInterval), 1)
    if RGBmode == 'bw' or RGBmode == 'seq':
        subPixN = x_oversample_factor
    elif RGBmode == 'simul':
        subPixN = 3 * x_oversample_factor
    else:
        raise Exception("Unrecognized RGB mode, cannot proceed")
    x_interval = pxlSize/subPixN  # space interval, in cm
    objPxlN = round(objSize/pxlSize)  # object size in pixels
    
    if objPxlN == 0:
        raise Exception("Object size too small for display resolution.")
    # convert spatial unit into degrees
    message = 'space interval: ' +\
        str(x_interval) + '%.4f' % x_interval, ' cm); time interval: ' + \
        str(t_interval) + 's'

    visualRange = math.ceil(10*recordingLength/pxlSize)*pxlSize

    t = np.arange(0, recordingLength, t_interval,
                  dtype=precision)  # time range
    # space range (in degrees)
    x = np.arange(0, visualRange, x_interval, dtype=precision)
    if RGBmode == 'bw':
        l_xt = np.zeros([len(x), len(t)], dtype=precision)
    elif RGBmode == 'seq' or RGBmode == 'simul':
        l_xt = np.zeros([len(x), len(t), 3], dtype=precision)
    t = t - recordingLength/2
    x = x - visualRange/2

    # Adjust cropping factor for displaying the stimulus and recon
    dbt = (t > disp_crop_bounds[0]).sum()
    dbx = (x > disp_crop_bounds[1]).sum()
    disp_start_inds = [dbt, dbx] # adjust factor to indices
    crop_generosity = 2 # portion of cropped part of image that is still generated (later to be cropped)
    c_idx = 0
    for t_idx in range(len(t)):
        if abs(t[t_idx]) >= disp_crop_bounds[0]*crop_generosity: # ignore most points that will be cropped by disp_factor
            continue
        xi = v*t[t_idx]
        if abs(xi) >= disp_crop_bounds[1]*crop_generosity: # ignore most points that are out of the visual range or will be cropped by disp_factor
            continue
        # find the index for object's current location.
        x_idx = findIndex(xi, x, force_np=True)
        if RGBmode == 'bw':
            l_xt[x_idx:x_idx+objPxlN*subPixN, t_idx] = contrast[0]
        else:
            l_xt[x_idx:x_idx+objPxlN*subPixN, t_idx, :] = contrast

    trackTime('Generated ideal stimulus.', 3)
    t_s = recordingLength*(np.arange(round(frmRate*recordingLength), dtype=precision)/round(frmRate*recordingLength)-0.5) # samples' time stamps

    if RGBmode == 'bw':
        l_xts = np.zeros([len(x), len(t)], dtype=precision)
    elif RGBmode == 'seq' or RGBmode == 'simul':
        l_xts = np.zeros([len(x), len(t), 3], dtype=precision)
    # normalize the continuous signal to the same RMS value as the sampled signal
    # index of the position vector of sampled points
    tt_idx = np.empty([len(t_s)], dtype=iprecision)
    xx_idx = np.empty([len(t_s)], dtype=iprecision)

    # line image at a single time point
    smp1D = np.empty(x.shape, dtype=precision)
    # line image equals to the number of pixels (each pixel contains pxlSize/x_interval sub pixels )
    pxl1D = np.empty(int(len(x)/x_oversample_factor), dtype=precision)

    for i in range(len(t_s)):

        # find the matched time stamp in the orignial signal
        tt_idx[i] = findIndex(t_s[i], t, force_np=True)

        if i < len(t_s)-1:  # the last sample point will not have an updated sample interval
            tt_idxNext = findIndex(t_s[i+1], t, force_np=True)
            # the number of sub-time points between every two frames
            smpInterval = tt_idxNext - tt_idx[i]
            flashInterval = int(smpInterval / nFlash)
            if RGBmode == 'seq':
                RGBInterval = round(holdInterval*flashInterval/3)
        # the sample point index
        xx_idx[i] = findIndex(v*t[tt_idx[i]], x, force_np=True)
        # the pixel index (sample point index divided by the number of subpixels per pixel)
        pxlVidx = int(xx_idx[i]/subPixN)
        if RGBmode == 'bw':
            pxl1D[:] = 0
            smp1D[:] = 0
            pxl1D[pxlVidx: min(len(pxl1D), pxlVidx + objPxlN)
                  ] = l_xt[xx_idx[i], tt_idx[i]]
            if antialiasing:
                pxl1D = ndimage.gaussian_filter1d(pxl1D, antialiasingF)
            # now fill in the values based on pixel fill factor.
            for j in range(len(pxl1D)):
                start, end = j*subPixN, j*subPixN + math.ceil(math.sqrt(fillF)*pxlSize/x_interval)
                smp1D[start:end] = pxl1D[j]
            l_xts[:, tt_idx[i]] = smp1D

            # Now fill in the values decided by hold interval.
            # @ 2022.09.13 this has to be changed for multi-flash protocol, i.e., presentation rate = n*frameRate, n = 1, 2, 3...
            initLine = smp1D.copy()
            # add the base value
            crrtLine = np.zeros(initLine.shape, dtype=precision)
            # iterate through flashes
            for flash_ind in range(nFlash):
                # the range of sample and hold (staircase function)
                start = tt_idx[i].item() + flash_ind*flashInterval
                duration = round(holdInterval*flashInterval)
                end = min(start + duration, len(t)-1)
                stairCaseRange = np.arange(
                    start, end, dtype=iprecision)
                # @2022916, Guanghan: now we need to add the temporal profile of pixels, when turned on and off
                weightT = np.ones(stairCaseRange.shape, dtype=precision)
                if pxlResponseT != 0:
                    cuttoffT = round(pxlResponseT/t_interval)
                    response_func = np.linspace(0, cuttoffT/pxlResponseT*t_interval, cuttoffT, endpoint=False, dtype=precision)
                    weightT[:cuttoffT] = response_func
                    weightT[-cuttoffT:] = response_func.copy()[::-1]

                # include the eye motion effect. The motion on the eye can be continuous, not limited by the pixel size, etc.
                for k in range(len(stairCaseRange)):
                    crrtShift = -1 * ve * t_interval*(k + flash_ind * flashInterval)
                    crrtIndx = findIndex(
                        crrtShift, x, force_np=True) - int(len(x)/2)
                    crrtLine[0:(len(x)+crrtIndx)
                             ] = initLine[max([0, -crrtIndx]):len(x)]
                    crrt_min = np.min(crrtLine)
                    fillVals = weightT[k]*(crrtLine - crrt_min) + crrt_min
                    fillInds = stairCaseRange[k]
                    l_xts[:, fillInds] = fillVals
        elif RGBmode == 'seq':
            for colorIdx in range(3):  # iterate through different colors
                smp1D[:] = 0
                pxl1D[:] = 0
                pxl1D[pxlVidx: min(len(pxl1D), pxlVidx + objPxlN)
                      ] = contrast[colorIdx]
                if antialiasing:
                    pxl1D = ndimage.gaussian_filter1d(pxl1D, antialiasingF)
                # now fill in the values based on pixel fill factor
                for j in range(len(pxl1D)):
                    start = j*subPixN
                    duration = math.ceil(math.sqrt(fillF)*x_oversample_factor)
                    smp1D[start:start + duration] = pxl1D[j]
                # Now fill in the values decided by hold interval.
                # @ 2022.09.13 this has to be changed for multi-flash protocol, i.e., presentation rate = n*frameRate, n = 1, 2, 3...
                initLine = smp1D.copy()
                # TODO OPTIMIZE
                initLine[initLine == min(initLine)] = 0
                # the original sample line at the start of this sample period,
                # matching the true values from the continuous signal
                crrtLine = np.zeros(
                    initLine.shape, dtype=precision)
                # the sample line shifted downward for -ve*t_interval (if there is an offset, the offset will be added)

                for flash_ind in range(nFlash):
                    start = tt_idx[i] + flash_ind * \
                        flashInterval + colorIdx*RGBInterval
                    end = min(start + RGBInterval, len(t) - 1)
                    stairCaseRange = np.arange(
                        int(start), int(end), dtype=iprecision)

                    if len(stairCaseRange) > 0:
                        weightT = np.ones(
                            stairCaseRange.shape, dtype=precision)
                        if pxlResponseT != 0:
                            cuttoffT = round(pxlResponseT/t_interval)
                            response_func = np.linspace(0, cuttoffT/pxlResponseT*t_interval, cuttoffT, endpoint=False, dtype=precision)
                            weightT[:cuttoffT] = response_func
                            weightT[-cuttoffT:] = response_func.copy()[::-1]
                        if spatialOffset:
                            offset = vx*t_interval*colorIdx*len(stairCaseRange)
                        else:
                            offset = 0
                        for k in range(len(stairCaseRange)):
                            if RGBmode == 'simul':
                                crrtShift = -ve*t_interval * \
                                    (k + flash_ind*flashInterval)
                            else:
                                crrtShift = -ve*t_interval * \
                                    (k + flash_ind*flashInterval +
                                     colorIdx*len(stairCaseRange))
                                crrtShift = crrtShift + offset
                            crrtIndx = findIndex(
                                crrtShift, x, force_np=True) - int(len(x)/2)
                            if crrtIndx < 0:
                                crrtLine[:len(
                                    x)+crrtIndx] = initLine[-crrtIndx:len(x)]
                            else:
                                crrtLine[crrtIndx:len(
                                    x)] = initLine[0:len(x)-crrtIndx]
                            l_xts[:, stairCaseRange[k], colorIdx] = weightT[k] * crrtLine
        else:
            for colorIdx in range(3):  # iterate through different colors
                smp1D[:] = 0
                pxl1D[:] = 0
                pxl1D[pxlVidx: min(len(pxl1D), pxlVidx + objPxlN)
                      ] = contrast[colorIdx]
                if antialiasing:
                    pxl1D = ndimage.gaussian_filter1d(pxl1D, antialiasingF)
                # now fill in the values based on pixel fill factor
                for j in range(len(pxl1D)):
                    start = j*subPixN + colorIdx*x_oversample_factor
                    duration = math.ceil(math.sqrt(fillF)*x_oversample_factor)
                    smp1D[start:start + duration] = pxl1D[j]
                # Now fill in the values decided by hold interval.
                # @ 2022.09.13 this has to be changed for multi-flash protocol, i.e., presentation rate = n*frameRate, n = 1, 2, 3...
                initLine = smp1D.copy()
                # TODO OPTIMIZE
                initLine[initLine == min(initLine)] = 0
                # the original sample line at the start of this sample period,
                # matching the true values from the continuous signal
                crrtLine = np.zeros(
                    initLine.shape, dtype=precision)
                # the sample line shifted downward for -ve*t_interval (if there is an offset, the offset will be added)

                for flash_ind in range(nFlash):
                    if RGBmode == 'simul':
                        start = tt_idx[i] + flash_ind*flashInterval
                        end = min(start + np.round(holdInterval *
                                  flashInterval), len(t) - 1)
                    else:
                        start = tt_idx[i] + flash_ind * \
                            flashInterval + colorIdx*RGBInterval
                        end = min(start + RGBInterval, len(t) - 1)
                    stairCaseRange = np.arange(
                        int(start), int(end), dtype=iprecision)

                    if len(stairCaseRange) > 0:
                        weightT = np.ones(
                            stairCaseRange.shape, dtype=precision)
                        if pxlResponseT != 0:
                            cuttoffT = round(pxlResponseT/t_interval)
                            response_func = np.linspace(0, cuttoffT/pxlResponseT*t_interval, cuttoffT, endpoint=False, dtype=precision)
                            weightT[:cuttoffT] = response_func
                            weightT[-cuttoffT:] = response_func.copy()[::-1]
                        if spatialOffset:
                            offset = vx*t_interval*colorIdx*len(stairCaseRange)
                        else:
                            offset = 0
                        for k in range(len(stairCaseRange)):
                            if RGBmode == 'simul':
                                crrtShift = -ve*t_interval * \
                                    (k + flash_ind*flashInterval)
                            else:
                                crrtShift = -ve*t_interval * \
                                    (k + flash_ind*flashInterval +
                                     colorIdx*len(stairCaseRange))
                                crrtShift = crrtShift + offset
                            crrtIndx = findIndex(
                                crrtShift, x, force_np=True) - int(len(x)/2)
                            if crrtIndx < 0:
                                crrtLine[:len(
                                    x)+crrtIndx] = initLine[-crrtIndx:len(x)]
                            else:
                                crrtLine[crrtIndx:len(
                                    x)] = initLine[0:len(x)-crrtIndx]
                            l_xts[:, stairCaseRange[k], colorIdx] = weightT[k] * crrtLine
    # track("%.2f seconds taken to configure the sampled stimulus..." %toc, 40)
    trackTime("Generated display stimulus.", 15)
    # now calculate the eye's CSF with the same dimension as the temporal spatial matrix
    l_xts = NP2CP(l_xts)
    l_xt = NP2CP(l_xt)
    t = NP2CP(t)
    x = asdeg(NP2CP(x))  # convert length to angle
    if gpu:
        trackTime("Moved to GPU.", 20)
    # convert length to angle.
    visualRange = asdeg(visualRange, True)
    # curve fit link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    def SpatialCSF(uu, Luminance):
        uu[abs(uu) < 1] = 1
        Su = 5200*cp.exp(-0.0016*((1+100/Luminance)**0.08)*uu**2)\
            / cp.sqrt((1+144/(objPxlN*pxlSize)**2 + 0.64*uu**2)*(63/Luminance**0.83 + 1/(1-cp.exp(-0.02*uu**2))))
        #empirical function of CSF (over spatial frequency):
        return Su

    def linearInterpoS(Su, uu, Luminance):
        fitx = cp.arange(2, 4, 0.05, dtype=precision)
        fity = SpatialCSF(fitx, Luminance)
        inds = abs(uu) < 1
        Su[inds] = cp.interp(uu[inds], fitx, fity)
        return Su
    
    def TemporalCSF(w, L):
        x = np.abs(w)
        x[x < 1] = 1
        numerator = L**2.51*5360*np.exp(-0.16*x**(L**-0.017))
        denominator = np.sqrt(
            (2.1e9*x**(9e-4*L)/L**-4.98+1)*(1.2e-7/L+1.96e-8/(1.007-np.exp(-2.7e-4*x**3.8)))-L**5
        )
        return numerator/denominator
    
    def linearInterpoT(Tw, ww, Luminance):
        fitx = cp.arange(2, 4, 0.05, dtype=precision)
        fity = TemporalCSF(fitx, Luminance)
        inds = abs(ww) < 1
        Tw[inds] = cp.interp(ww[inds], fitx, fity)
        return Tw
    '''
        our knowns are L_max, LR_max + LB_max + LG_max, contrastR, contrastG, contrastB
        our unknowns are LR_min, LG_min, LB_min
        L_max/3 = LR_max = LB_max = LG_max
        LR_min = LR_max/(1+contrastR) | weber contrast formula
        LG_min = LG_max/(1+contrastG) | weber contrast formula
        LB_min = LB_max/(1+contrastB) | weber contrast formula
    '''
    L_max = luminance # luminance of stimulus is given by user
    L_b = sum([(L_max/3) / (1 + c) for c in contrast]) # background luminance
    fovea_area = (FOVEA_SIZE/2)**2 * math.pi # area of fovea
    objArea = asdeg(objPxlN*pxlSize) * FOVEA_SIZE # area of object
    Luminance = (objArea * L_max + (fovea_area - objArea) * L_b) / fovea_area # average luminance on fovea
    # define the spectrum vectors as below to calculate the CSF of the eye
    Ft = len(t)/recordingLength
    Fx = len(x)/visualRange
    ft = cp.linspace(-Ft/2, Ft/2, len(t), endpoint=False)
    fx = cp.linspace(-Fx/2, Fx/2, len(x), endpoint=False)
    print("ft: ", Ft/2, "fx: ", Fx/2)
    print("ft range: ", ft[0], ft[-1], "fx range: ", fx[0], fx[-1])
    print("ft length: ", len(ft), "fx length: ", len(fx))
    # in order to speed up the calculation, we will not apply the CSF function to the entire spatial temporal domain.
    # values outside the boundary defined by fxlim and ftlim will be set to zero.
    fxlim = 120
    ftlim = 1000

    max_ind_t = math.ceil(len(t)/2) - round(ftlim * recordingLength)
    max_ind_x = math.ceil(len(x)/2) - round(fxlim * visualRange)
    start_ind_ft, start_ind_fx = max(max_ind_t, 0), max(max_ind_x, 0)
    print("start_ind_ft: ", start_ind_ft, "start_ind_fx: ", start_ind_fx)
    end_ind_ft = -start_ind_ft if start_ind_ft > 0 else None
    end_ind_fx = -start_ind_fx if start_ind_fx > 0 else None
    ft_short = ft[start_ind_ft:end_ind_ft]
    fx_short = fx[start_ind_fx:end_ind_fx]
    Su = (linearInterpoS(SpatialCSF(fx_short, Luminance),
          fx_short, Luminance) ** 0.5)[..., None]
    Tw = (linearInterpoT(TemporalCSF(ft_short, Luminance),
          ft_short, Luminance) ** 0.5)[None, ...]
    filter_eye_pad = [(math.floor(int(len(x) - len(fx_short))/2), math.ceil(int(len(x) - len(fx_short))/2)),
                      (math.floor(int(len(t) - len(ft_short))/2), math.ceil(int(len(t) - len(ft_short))/2))]
    Su = cp.pad(Su, [filter_eye_pad[0], (0, 0)])
    Tw = cp.pad(Tw, [(0, 0), filter_eye_pad[1]])
    trackTime("Calculated spatiotemporal CSF.", 25)
    def norm(inputImg, clip=False, postprocess=False):
        if clip:
            vmin = inputImg.min(axis=(0, 1))
            if inputImg.ndim == 2:
                second_max = cp.partition(inputImg, -2, axis=None)[-2]
                inputImg[inputImg > second_max] = second_max
                inputImg[inputImg == vmin] = cp.round(vmin, 4)
            else:
                for vind, vmin_val in enumerate(vmin):
                    second_max = cp.partition(inputImg[:, :, vind], -2, axis=None)[-2]
                    img = inputImg[:, :, vind]
                    img[img > second_max] = second_max
                    img[img == vmin_val] = cp.round(vmin_val, 4)
            outputImg = inputImg - cp.min(inputImg)
            logimg = outputImg[outputImg > outputImg.mean()]
            mn, sd = logimg.mean(), logimg.std()
            outputImg = cp.clip(outputImg, 0, mn + sd*3)

            return inputImg
        elif postprocess:
            if inputImg.ndim == 3:
                inputImg = inputImg.mean(-1)
            return inputImg / cp.max(inputImg)
        else:
            outputImg = inputImg - cp.min(inputImg)
            if cp.max(outputImg) == 0:
                print("WARNING: Max of outputImg is 0.", cp.max(outputImg), cp.min(outputImg), cp.max(inputImg), cp.min(inputImg))
            out = outputImg / cp.max(outputImg) * max(contrast)
            return out
    if RGBmode == 'bw':
        f_complex_c = FFTSHIFT(FFT2(l_xt))
        f_complex = FFTSHIFT(FFT2(l_xts))
        trackTime("Calculated spectrum.", 40)
        f_result_c = norm(cp.abs(f_complex_c), clip=True)
        f_result = norm(cp.abs(f_complex), clip=True)
        f_complex_c = f_complex_c * Tw * Su
        f_complex = f_complex * Tw * Su
        f_filtered_c = norm(cp.abs(f_complex_c), clip=True)
        f_filtered = norm(cp.abs(f_complex), clip=True)
        trackTime("Enforced visual system model.", 55)
        f_recon_c = cp.abs(IFFT2(FFTSHIFT(f_complex_c)))
        f_recon = cp.abs(IFFT2(FFTSHIFT(f_complex)))
        trackTime("Calculated reconstruction for artifact predictions.", 70)
    else:
        f_complex_c = cp.empty(l_xt.shape[:2], dtype="complex128")
        f_complex = cp.empty(l_xts.shape[:2], dtype="complex128")
        f_result_c = cp.empty(l_xt.shape, dtype=precision)
        f_result = cp.empty(l_xts.shape, dtype=precision)
        f_filtered_c = cp.zeros(l_xt.shape, dtype=precision)
        f_filtered = cp.zeros(l_xts.shape, dtype=precision)
        f_recon_c = cp.empty(l_xt.shape, dtype=precision)
        f_recon = cp.empty(l_xts.shape, dtype=precision)
        chnName = ['R', 'G', 'B']
        progress = 30
        pincrement = 5
        for jj in range(len(contrast)):
            f_complex_c = FFTSHIFT(FFT2(l_xt[:, :, jj]))
            f_complex = FFTSHIFT(FFT2(l_xts[:, :, jj]))
            message = "Calculated spectrum - channel %s." % chnName[jj]
            trackTime(message, progress)
            progress += pincrement
            f_result_c[:, :, jj] = norm(cp.abs(f_complex_c), clip=True)
            f_result[:, :, jj] = norm(cp.abs(f_complex), clip=True)
            f_complex_c = f_complex_c * Tw * Su
            f_complex = f_complex * Tw * Su
            f_filtered_c[:, :, jj] = norm(cp.abs(f_complex_c), clip=True)
            f_filtered[:, :, jj] = norm(cp.abs(f_complex), clip=True)
            trackTime("Enforced visual system model - channel %s." %
                      chnName[jj], progress)
            progress += pincrement
            f_recon_c[:, :, jj] = cp.abs(IFFT2(FFTSHIFT(f_complex_c)))
            f_recon[:, :, jj] = cp.abs(IFFT2(FFTSHIFT(f_complex)))
            trackTime("Calculated reconstruction for artifact predictions - channel %s." %
                      chnName[jj], progress)
            progress += pincrement
    del f_complex_c, f_complex
    f_filtered = norm(f_filtered, postprocess=True)
    f_filtered_c = norm(f_filtered_c, postprocess=True)
    f_result = norm(f_result, postprocess=True)
    f_result_c = norm(f_result_c, postprocess=True)
    l_xt = norm(l_xt)
    l_xts = norm(l_xts)
    f_recon_c = norm(f_recon_c)
    f_recon = norm(f_recon)
    trackTime("Scaled data for plotting.", 73)
    energy_c, energy = cp.sum(f_recon_c**2), cp.sum(f_recon**2)
    f_recon = cp.clip(f_recon * (energy_c/energy)**0.5, 0, 1)
    trackTime("Normalized energy for plotting.", 75)
    
    end_bound_t, end_bound_x = disp_crop_bounds[0], asdeg(disp_crop_bounds[1])
    start_ind_t, start_ind_x = disp_start_inds
    end_ind_t, end_ind_x = [-i if i != 0 else None for i in disp_start_inds]
    t = t[start_ind_t:end_ind_t]
    x = x[start_ind_x:end_ind_x]
    recordingLength -= 2*end_bound_t
    visualRange -= 2*end_bound_x
    l_xt = l_xt[start_ind_x:end_ind_x, start_ind_t:end_ind_t]
    l_xts = l_xts[start_ind_x:end_ind_x, start_ind_t:end_ind_t]
    f_recon_c = f_recon_c[start_ind_x:end_ind_x, start_ind_t:end_ind_t]
    f_recon = f_recon[start_ind_x:end_ind_x, start_ind_t:end_ind_t]
    
    # Crop frequency spectra
    pad_fx, pad_ft = 0.2, 0.2
    pad_inds_ft, pad_inds_fx = int((len(ft)//2-start_ind_ft)*pad_ft), int((len(fx)//2-start_ind_fx)*pad_fx)
    if True:
        pad_inds_ft, pad_inds_fx = -400, 0
    start_ind_ft, start_ind_fx = max(start_ind_ft - pad_inds_ft, 0), max(start_ind_fx - pad_inds_fx, 0)
    end_ind_ft = -start_ind_ft if start_ind_ft else None
    end_ind_fx = -start_ind_fx if start_ind_fx else None
    f_result_c = f_result_c[start_ind_fx:end_ind_fx, start_ind_ft:end_ind_ft]
    f_result = f_result[start_ind_fx:end_ind_fx, start_ind_ft:end_ind_ft]
    f_filtered_c = f_filtered_c[start_ind_fx:end_ind_fx, start_ind_ft:end_ind_ft]
    f_filtered = f_filtered[start_ind_fx:end_ind_fx, start_ind_ft:end_ind_ft]
    ft = ft[start_ind_ft:end_ind_ft]
    fx = fx[start_ind_fx:end_ind_fx]

    # this block is to generate the output figures
    xlabel_txtt = 'Time (s)'
    ylabel_txts = 'Spatial contrast on retina (degree)'
    dispRangets = [cp.min(t).item(), cp.max(t).item(),
                   cp.min(x).item(), cp.max(x).item()]

    xlabel_txtf = 'Temporal Frequency on retina (Hz)'
    ylabel_txtf = 'Spatial Frequency on retina(cycles/degree)'
    dispRangeF = [cp.min(ft).item(), cp.max(ft).item(),
                  cp.min(fx).item(), cp.max(fx).item()] 

    if gpu:
        l_xt = CP2NP(l_xt)
        l_xts = CP2NP(l_xts)
        f_recon_c = CP2NP(f_recon_c)
        f_recon = CP2NP(f_recon)
        f_result_c = CP2NP(f_result_c)
        f_result = CP2NP(f_result)
        f_filtered_c = CP2NP(f_filtered_c)
        f_filtered = CP2NP(f_filtered)
        t = CP2NP(t)
        x = CP2NP(x)
        visualRange = CP2NP(visualRange)
        trackTime("Moved to CPU.", 80)

    fig10 = plt.figure(figsize=(50, 25))
    gs = gridspec.GridSpec(2, 4, figure=fig10)

    fig10.add_subplot(gs[0, 0])
    compressPlot(l_xt, 1, 0, recordingLength, visualRange, t, x, dispRangets,
                    xlabel_txtt, ylabel_txts, 'Continuous Stimulus ', vrange=(0, 1))
    fig10.add_subplot(gs[0, 1])
    compressPlot(f_result_c, 1, 0, Ft, Fx, ft, fx, dispRangeF,
                    xlabel_txtf, ylabel_txtf, 'Continuous Stimulus frequency spectrum')
    fig10.add_subplot(gs[0, 2])
    compressPlot(f_filtered_c, 1, 0, Ft, Fx, ft, fx, dispRangeF,
                    xlabel_txtf, ylabel_txtf, 'Continuous stimulus filtered spectrum')
    fig10.add_subplot(gs[0, 3])
    compressPlot(f_recon_c, 1, 0, recordingLength, visualRange, t, x, dispRangets,
                    xlabel_txtt, ylabel_txts, 'Perceived continuous stimulus', vrange=(0, 1))
    trackTime("Plotted continuous stimulus.", 85)
    fig10.add_subplot(gs[1, 0])
    compressPlot(l_xts, 1, 0, recordingLength, visualRange, t, x, dispRangets,
                    xlabel_txtt, ylabel_txts, 'Sampled Stimulus ' + strUpdate, vrange=(0, 1))
    fig10.add_subplot(gs[1, 1])
    compressPlot(f_result, 1, 0, Ft, Fx, ft, fx, dispRangeF,
                    xlabel_txtf, ylabel_txtf, 'Sampled Stimulus frequency spectrum')
    fig10.add_subplot(gs[1, 2])
    compressPlot(f_filtered, 1, 0, Ft, Fx, ft, fx, dispRangeF,
                    xlabel_txtf, ylabel_txtf, 'Sampled stimulus filtered spectrum')
    fig10.add_subplot(gs[1, 3])
    compressPlot(f_recon, 1, 0, recordingLength, visualRange, t, x, dispRangets,
                    xlabel_txtt, ylabel_txts, 'Perceived sampled stimulus', vrange=(0, 1))
    plt.tight_layout()
    trackTime("Plotted display stimulus.", 90)

    if savePath:
        fig10.savefig(savePath)
        if tracker.compareInd != None and tracker.pauseCompare != True:
            img = io.imread(savePath)
            if tracker.compareInd == 1:
                io.imsave("compare0.png", img[:1270, :])
            io.imsave("compare%d.png" % tracker.compareInd, img[1260:, :])
            tracker.compareInd += 1
            trackTime("Generated compare figure.", 93)

    trackTime("Finished plotting.", 95)
    total_time = time.time()-init_time
    track('Completed execution. (total %.1fs)' % total_time, 100)
    plt.close('all')
    gc.collect()
    mem_clear()
    return total_time


def stereo_parameters(frmRate=60, holdInterval=1, nFlash=1, capture='alt', vx=10, ve=True, viewing_D=50, track=print, tracker=None, **kwargs):
    start_time = time.time()
    # depth distortion in stereo display. Output: disparity in unit of degree
    # Assume the order of eye stimulation is parallel to the object's motion. i.e.,if object is moving from left to right,
    # the left eye is stimulated first, then the right eye; If anti-parallel instead, the sign of disparity should be flipped
    # display parameters
    # capture = 'alt' #alternating capturing (alt), or simultaneous capturing (simul)
    # frmRate = 50
    # holdInterval = 0.5
    # nFlash = 3 # the times of repitition for each frame; frmRate*nFlash = presentation rate
    # # pixel hold interval, range of 0-1 (e.g. if same value is held until
    # # next sample point, then holdInterval equals to 1)
    # vx = 10 # cm/s

    # viewing parameters
    # viewing_D = 50 # cm
    # ve = vx #0 # eye motion speed, cm/s
    ve = vx if ve else 0
    frmRate = 2*frmRate if capture == 'alt' else frmRate
    # ---------------end of user input
    t_interval = 0.0001
    print('t_interval = %.2f ms' % (t_interval*1000))

    track('Preparing parameters.', 10)

    # number of time stamps among flashes
    flashInterval = 1/frmRate/nFlash/t_interval
    halfInterval = int(flashInterval/2)
    halfHoldInterval = round(halfInterval*holdInterval)
    print('halfInterval = %d steps' % halfInterval,
          'flashInterval = %d steps' % flashInterval)

    recordingLength = 5  # this is going to be fixed for stereo display
    displayLength = 0.2
    total_num_flashes = np.ceil(recordingLength * frmRate * nFlash)
    recordingLength = total_num_flashes * flashInterval * \
        t_interval  # correct for convenience
    print('recordingLength = %.2f s' % recordingLength)

    # convert velocity into degrees/s
    vx = math.atan(vx / viewing_D)/math.pi*180
    ve = math.atan(ve / viewing_D)/math.pi*180

    t = np.linspace(0, recordingLength, int(recordingLength/t_interval), endpoint=False)
    # these two vectors are for the plotting purpose only.
    # The gaps between each pair of flashs are left as NaN so they are not connected during plotting
    pos1 = np.full(t.shape, np.nan)
    pos2 = np.full(t.shape, np.nan)
    # generate the sampled signal
    t_s = np.linspace(0, recordingLength, int(frmRate*recordingLength), endpoint=False) # samples' time stamps
    track("Computing.", 25)
    # if the frames for two eyes are captured simultaneously
    if capture == 'sim':
        # now create the time stamps for left and right eyes using nFlash
        t_s1 = np.linspace(0, recordingLength, int(frmRate*recordingLength*nFlash), endpoint=False)
        pos_s1 = np.zeros(t_s1.shape)
        # time stamps for the eye stimulated first.
        # time stamps for the second eye will have a shift half the flash interval
        t_s2 = t_s1 + halfInterval*t_interval
        pos_s2 = np.zeros(t_s2.shape)
        t_s12 = np.linspace(0, recordingLength, int(frmRate*recordingLength*nFlash*2), endpoint=False)
        pos_s12 = np.zeros(t_s12.shape)
        # time stamps for both eyes.
        for ind, tstamp in enumerate(t_s):
            # find the matched time stamp in the orignial signal
            tt_idx = findIdx(tstamp, t)
            for flash_ind in range(nFlash):  # iterate through flashes
                # initial location of the stimulus at the first time stamp in this flash cycle
                pos_init = (vx - ve)*tstamp - ve * \
                    flash_ind * (1/frmRate/nFlash)
                # assign the position to the first eye
                pos_s1[nFlash*ind + flash_ind] = pos_init
                # assign the position to the seond eye plus the shift due to eye motion
                pos_s2[nFlash*ind + flash_ind] = pos_init - \
                    ve * (1/frmRate/nFlash/2)
                # now assign the staircase values due to non-zero 'holdInterval'
                init_idx = math.floor(tt_idx + flash_ind * flashInterval)
                duration = -ve*t_interval*halfHoldInterval
                nsteps = halfHoldInterval
                start1 = pos_init
                start2 = start1 - ve*t_interval*halfInterval
                pos1[init_idx:init_idx + halfHoldInterval] = np.linspace(start1, start1+duration, nsteps, endpoint=False)
                pos2[init_idx + halfInterval: init_idx + halfInterval + halfHoldInterval] = np.linspace(start2, start2+duration, nsteps, endpoint=False)

        # now combine position matrices from both eyes
        pos_s12[np.arange(0, len(pos_s12), 2)] = pos_s1
        pos_s12[np.arange(1, len(pos_s12), 2)] = pos_s2
        disparity_seq = pos_s12[1:] - pos_s12[:-1]

    # if the frames for two eyes are captured separately, no multi-flash protocols will be considered.
    else:
        # sampled stimulus locations on the retina
        pos_s = (vx - ve)*t_s
        disparity_seq = pos_s[1:] - pos_s[:-1]

        # generate matrices for plotting (separte the stimulus perceived by the first and second eyes)
        t_s12 = t_s
        t_s1 = t_s12[np.arange(0, len(pos_s)-1, 2)]
        t_s2 = t_s12[np.arange(1, len(pos_s), 2)]
        pos_s1 = pos_s[np.arange(0, len(pos_s)-1, 2)]
        pos_s2 = pos_s[np.arange(1, len(pos_s), 2)]
        # to plot the stimulus correctly:
        nsteps = round(holdInterval/frmRate/t_interval)
        for ind, tstamp in enumerate(t_s):
            # find the matched time stamp in the orignial signal
            tt_idx = findIdx(tstamp, t)
            start, duration = t[tt_idx]*(vx - ve), -ve*t_interval*nsteps
            pos_holdinterval = np.linspace(start, start+duration, nsteps, endpoint=False)
            if (ind % 2) == 0:
                pos1[tt_idx:tt_idx + nsteps] = pos_holdinterval
            else:
                pos2[tt_idx:tt_idx + nsteps] = pos_holdinterval

    # since the disparity should always be calculated as 'left - right', the signs of entries with odd indices
    # in the matrix above should be flipped
    disparity_seq[::2] *= -1
    track("Finalizing plots.", 75)
    # calculate the average disparity
    avg_disparity_seq = np.mean(disparity_seq)
    if abs(avg_disparity_seq) < 0.005:
        avg_disparity_seq = 0

    # now plot the stimulus and the disparity
    fig11 = plt.figure(figsize=(8, 10))
    nrows = 2
    ncols = 1
    lbFontSize = 16
    ttlFontSize = 20
    txtFontSize = 20
    tickFontSize = 14
    gs = gridspec.GridSpec(nrows, ncols, figure=fig11)
    fig11.add_subplot(gs[0, 0])
    plt.plot(t[t < displayLength], pos1[t < displayLength],
             color='r', linewidth=1)
    plt.plot(t[t < displayLength], pos2[t < displayLength],
             color='b', linewidth=1)
    plt.gca().legend(['Left', 'Right'], fontsize=txtFontSize-2)
    
    plt.xticks(np.arange(0, 0.35, 0.1), [])
    plt.yticks(np.arange(-1, 5, 1), fontsize=tickFontSize)
    plt.gca().set_ylim([-1.4, 4.2])
    plt.gca().set_xlim([-0.01, displayLength + 0.01])
    ylabeltxt = 'Stimulus location\n on retina (degree)'
    plt.ylabel(ylabeltxt, fontsize=lbFontSize)
    if capture == 'sim':
        titletxt = '$Stimulus: C_{%s}P_{alt%dX} $' % (capture, nFlash)
    else:
        titletxt = '$Stimulus: C_{%s}P_{alt1X}$' % capture
    plt.title(titletxt, fontsize=ttlFontSize)

    fig11.add_subplot(gs[1, 0])
    plt.plot(t_s12[0:-1][t_s12[0:-1]<=displayLength], disparity_seq[t_s12[0:-1]<=displayLength], color='g',
             linestyle='none', marker='o', markersize=5)
    plt.plot([-0.01, displayLength + 0.01], [0, 0],
             color=[0.8, 0.8, 0.8], linestyle='-')
    line, = plt.plot([0, displayLength], [avg_disparity_seq,
             avg_disparity_seq], color='k', linestyle='--')
    avg_disparity_seq_arcsec = round(avg_disparity_seq*60*60)
    plt.gca().legend([line], ["Avg disparity: %d″" % avg_disparity_seq_arcsec], fontsize=txtFontSize-2, loc='upper left')
    plt.yticks(np.arange(-0.2, 0.5, 0.2), fontsize=tickFontSize)
    print("Average disparity: %.2f" % avg_disparity_seq)
    xlabeltxt = 'Time (s)'
    ylabeltxt = 'Disparity (degree)'
    titletxt = '$Perceived\ disparity\ over\ time$'
    plt.xlabel(xlabeltxt, fontsize=lbFontSize)
    plt.ylabel(ylabeltxt, fontsize=lbFontSize)
    plt.title(titletxt, fontsize=ttlFontSize)
    plt.xticks(np.arange(0, displayLength+ 0.01, 0.1), fontsize=tickFontSize)
    plt.gca().set_ylim([-0.4, 0.5])
    plt.gca().set_xlim([-0.01, displayLength + 0.01])
    print(nFlash, "Flashes")
    fig11.savefig("run_output.png")
    if tracker and tracker.stereoCompareInd is not None and not tracker.stereoPauseCompare:
        if tracker.stereoCompareInd != 0:
            img = io.imread("run_output.png")
            img[:img.shape[0]//2, :60] = img[0,0]
            io.imsave(f"stereoCompare{tracker.stereoCompareInd}.png", img[:, 50:])
        else:
            fig11.savefig("stereoCompare0.png")
        print("Saved stereoCompare image", tracker.stereoCompareInd)
        tracker.stereoCompareInd += 1
        
    track('Completed execution. (total %.1fs)' % (time.time()-start_time), 100)