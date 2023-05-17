'''
    Main file for running the backend server of DMAPPE.
'''
import traceback
import json
import gc
import sys
from getopt import getopt
from base64 import b64encode
from flask import Flask, jsonify, request
from classes import ToolboxRunner
from numpy import concatenate
from skimage.io import imread, imsave
from waitress import serve

device_info = ["Device 0: CPU"]
try:
    import cupy  # Test if cupy is available
    import GPUtil
    for i, gpu in enumerate(GPUtil.getGPUs()):
        device_info.append('Device {:d}: GPU ({:.1f}GB)'.format(
            i+1, gpu.memoryTotal/1000))
except Exception as e:
    print("GPU Exception: " + str(e))


# Initialize app and objects
app = Flask(__name__, static_url_path='/static')
runner = ToolboxRunner()

def send_image(imname, imname2=None):
    '''
        Convert image to JSON and pack for sending to UI.
    '''
    a = "false"
    b = "false"
    with open(imname, "rb") as image:
        r = image.read()
        a = b64encode(r).decode("utf-8")

    if imname2 is not None:
        with open(imname2, "rb") as image:
            r = image.read()
            b = b64encode(r).decode("utf-8")

    return jsonify({'status': True, 'image': a, 'compare': b})


@app.route('/poll')
def poll():
    '''
        Respond to polling requests from UI.
    '''
    return jsonify({'connected': 'true'})

@app.route('/getinfo')
def get_device_info():
    '''
        Respond to a device info request from UI.
    '''
    print("INFO SENT:", device_info)
    return jsonify({'device_info': device_info})

@app.route('/setdevice', methods=['POST'])
def set_device_info():
    '''
        Set device for run.
    '''
    try:
        jsdata = request.form  # immutable dict with key=data, val=''
        device = int(jsdata.to_dict()['device'])
        runner.tracker.device = device
        print("DEVICE SET TO:", device)
        return jsonify({'status': 'true'})
    except Exception as exc:
        print(f"\n\nDevice Set Failure: {exc}\n\n")
        return {}


@app.after_request
def add_header(response):
    '''
        Modify outgoing response headers.
    '''
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/reset')
def reset():
    '''
        Force reset of UI.
    '''
    runner.tracker.resetCompare()
    runner.key_mode = False
    print("RESET COMPARE")
    return jsonify({'cleared': 'true'})

@app.route('/test') #TODO Change to /getimage
def test():
    '''
        Send image to UI.
    '''
    print("Compare inds", runner.tracker.compareInd, runner.tracker.stereoCompareInd)
    compare = [not runner.tracker.pauseCompare and runner.tracker.compareInd > 2,
               not runner.tracker.pauseStereoCompare and runner.tracker.stereoCompareInd > 1]
    try:
        if not any(compare):
            return send_image("run_output.png")
        if compare[0]:
            new_img = concatenate([imread("compare%d.png" % i)
                                  for i in range(runner.tracker.compareInd)])
            imsave("compare.png", new_img)
            return send_image("compare.png", "run_output.png")
        if compare[1]:
            print(runner.tracker.stereoCompareInd, "images in Stereo Compare")
            new_img = concatenate([imread("stereoCompare%d.png" % i)[
                                  :, :-65] for i in range(runner.tracker.stereoCompareInd)], axis=1)
            imsave("stereoCompare.png", new_img)
            return send_image("stereoCompare.png", "run_output.png")
    except Exception as exc:
        print(f"\n\nFAILED COMPARISON FEATURE: {exc}\n\n")
        return send_image("run_output.png")


# Send current progress.
@app.route('/getprog')
def progress():
    '''
        Handle UI poll for progress during runs.
    '''
    # message, prog = runner.getProgress()
    prog = runner.getProgress()
    out = {
        str(i): {
            "progress": prog[i][1],
            "message": prog[i][0]
        } for i in range(len(prog))
    }
    out["length"] = len(prog)
    # out["length"] = len(prog)
    # print('Sent progress: ', message, prog)
    return jsonify(out)#'progress': prog, 'message': message})

# Receive parameters for run, and start run.


@app.route('/postparams', methods=['POST'])
def get_post_javascript_data():
    '''
        Handle UI run request.
    '''
    print('Attemped to run')
    try:
        gc.collect()
        jsdata = request.form  # immutable dict with key=data, val=''
        print(str(jsdata.to_dict()))
        stereo = str(jsdata.to_dict()['stereo']) == 'true'
        compare = str(jsdata.to_dict()['compare']) == 'true'
        key = str(jsdata.to_dict()['key']) == 'true'
        if key and stereo and compare:
            runner.key_mode = True
            runner.tracker.stereoCompareInd = None
        elif runner.key_mode and stereo and compare:
            runner.tracker.stereoCompareInd = 1
        runner.tracker.pauseStereoCompare = not stereo or not compare
        runner.tracker.pauseCompare = stereo or not compare
        print("COMPARE:", compare, "STEREO:", stereo)
        print("PAUSE COMPARE:", runner.tracker.pauseCompare,
              "PAUSE STEREO COMPARE:", runner.tracker.pauseStereoCompare)
        if (stereo and compare) and runner.tracker.stereoCompareInd is None:
            runner.tracker.stereoCompareInd = 0
            print("STEREO COMPARE STARTED")
        elif (not stereo and compare) and runner.tracker.compareInd is None:
            runner.tracker.compareInd = 1
            print("REGULAR COMPARE STARTED")
        # get the string of python dict version
        jsdata = str(jsdata.to_dict()['params'])
        dict_data = json.loads(jsdata)  # convert from string to dict
        if isinstance(dict_data, dict):
            runner.updateParams(dict_data)
            print("\n\nSuccessful Params Acquisition", jsdata, "\n\n")
            print(stereo)
            runner.run(stereo=stereo)
        else:
            print("\n\nFailed Params Acquisition", jsdata, "\n\n")
        return dict_data
    except Exception as exc:
        print(f"\n\nFAILED RUN: {exc}\n\n")
        print(traceback.format_exc())
        return {}


if __name__ == '__main__':
    if len(sys.argv) > 1 and str(sys.argv[1]) in ["--dev", "-d"]:
        print("DEV MODE")
        app.run(debug=True, host='0.0.0.0', port=3334)
    else:
        serve(app, listen='0.0.0.0:3334')
    # app.run(debug=True, host='0.0.0.0', port=3334)
    # serve(app, host='0.0.0.0', port=5000)
