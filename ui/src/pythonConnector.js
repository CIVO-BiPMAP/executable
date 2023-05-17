import './app.css';
import {CircularProgressWithLabel} from "./ProgressWithLabel";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import React from "react";
import {Divider, ToggleButton, ToggleButtonGroup, Grid, TextField} from '@mui/material';
import $ from 'jquery';
import Typography from "@mui/material/Typography";
import {theme, server} from './index';
import download from "downloadjs";
import downIcon from "./assets/down.png";
import compIcon from "./assets/compare.png";
import keyIcon from "./assets/key.png";
import DeviceMenu from './menu';
import { documentationString, ParameterControl, handleMenuChange, getStereo, getParameterDict } from './app';

const stereo_params = {
  "frmRate": 60,
  "ve": false,
}

function FileNameInput(props) {
  const [title, setTitle] = React.useState(props.value);
  props.setValue(title);
  return (
    <TextField
      label="Title"
      variant="outlined"
      size='small'
      color="primary"
      style={{maxWidth: '100', height: '100%'}}
      value={title}
      onChange={event=>{
        setTitle(event.target.value)
        props.setValue(event.target.value)
        window.localStorage.setItem("title", event.target.value)
      }} />
  )
}

export default function Content(props) {
  const progress_update_delay = 100;
  const [source, setSource] = React.useState();
  const [compareSource, setCompareSource] = React.useState();
  const [progress, setProgress] = React.useState(0);
  const [message, setMessage] = React.useState(documentationString);
  const [compare, setCompare] = React.useState([]);
  const [key, setKey] = React.useState([]);
  const [stereo, setStereo] = React.useState(getStereo());
  const [stereoLabel, setStereoLabel] = React.useState(stereo ? "2D Menu" : "Stereo Menu");
  const [isRunDone, setIsRunDone] = React.useState(false);
  const [deviceInfo, setDeviceInfo] = React.useState([]);
  let interval = null;
  let title = window.localStorage.getItem("title") ?? '';

  function handleStereoChange() {
    if (stereo) {
      setKey([]);
    } else {
      const param = getParameterDict()
      for (const [key, value] of Object.entries(stereo_params)) {
        param[key] = value;
      }
      window.localStorage.setItem("params", JSON.stringify(param));
      handleKeyToggle(null, [true])
    }
    let newVal = !stereo //!stereo;
    window.localStorage.setItem("stereo", newVal);
    setStereo(newVal);
    setStereoLabel(newVal ? "2D Menu" : "Stereo Menu");
  }

  if (deviceInfo.length === 0) {
    fetch(
      server('/getinfo')).then(
        res => res.json()
      ).then(res => {
        console.log(res.device_info);
        if (res.device_info) {
          setDeviceInfo(res.device_info);
        }
      }
    );
  }
  
  function messageUpdate(text) {
    setMessage(message => text + '\n' + message)
  }

  function messageReset(text="") {
    setMessage(text)
  }

  async function initRun() {
    setIsRunDone(false);
    clearInterval(interval);
    setSource('')
    setProgress(0)
    messageReset("Initializing run.")
    console.log({ 
      'params': window.localStorage.getItem("params") || '{}',
      'compare': compare.length !== 0,
      'stereo': stereo,
      'key': key
    })
    await $.post(server('/postparams'), { 
      'params': window.localStorage.getItem("params") || '{}',
      'compare': compare.length !== 0,
      'stereo': stereo,
      'key': key.length !== 0
    });
    setKey([]);
  }

  async function requestImage() {
    fetch(server('/test')).then(res => res.json()).then(data => {
      if (data.image) {
        setSource('data:image/jpg;base64,' + data.image)
      }
      messageUpdate("Ready to initialize run.")
      setIsRunDone(true);
      if (data.compare !== "false") {
        setCompareSource('data:image/jpg;base64,' + data.compare)
      } else {
        setCompareSource(undefined)
      }
    })
  }

  function resetInterval() {
    clearInterval(interval);
    setProgress(0);
  }

  function errorBehavior(data) {
    console.log('Caught error: ' + data.message)
    messageUpdate('Caught error: ' + data.message)
    resetInterval();
  }

  function expectedBehavior(run, data) {
    run.nopCount = 0;
    messageUpdate(data.message)
    setProgress(data.progress)
    run.asyncProgress = data.progress
    if (data.progress === 100) {
      clearInterval(interval)
      requestImage();
    } 
  }

  async function poll(run, delay) {
    fetch(server('/getprog')).then(res => res.json()).then(prog => {
      if (prog.length) {
        for (let i = 0; i < prog.length; i++) {
          let data = prog[i];
          if (data.message === undefined || data.progress === undefined || data.progress === -1) {
            errorBehavior(data);
          } else if (data.progress !== 'nop' && (data.progress > run.asyncProgress || run.asyncProgress === 0)) {
            expectedBehavior(run, data);
          } else if (run.nopCount*delay/1000 > 60) {
            console.log('NOP count ', run.nopCount, 'Progress ', run.asyncProgress)
            resetInterval();
          } else {
            run.nopCount += 1;
          }
        }
      }
    })
  }

  async function run() {
    if (interval !== null) {
      return null;
    }
    await initRun();
    const run_vars = {
      'asyncProgress': 0,
      'nopCount': 0
    }
    interval = setInterval(() => poll(run_vars, progress_update_delay), progress_update_delay);
  };

  function handleCompareToggle(_event, newValue) {
    setCompare(newValue)
  }

  function handleKeyToggle(_event, newValue) {
    setKey(newValue)
    if (newValue.length !== 0) {
      handleCompareToggle(null, newValue);
    }
  }

  function MainContent() {
    if (isRunDone && source) {
      return (
        <div style={{height: '70vh', marginBottom: 5}}>
          <img id="img" alt="" src={source} style={{maxWidth: '100%', maxHeight: '100%', height: 'auto', width: 'auto'}}/>
        </div>
      );
    }
    return <CircularProgressWithLabel value={progress} sx={{margin: 'auto'}}/>;
  }

  function ButtonStack() {
    return (
      <Stack direction="row" spacing={1} >
        <Button variant="outlined" color="primary" label={stereoLabel} onClick={handleStereoChange} sx={{width: 144, height: '100%'}}>
          {stereoLabel}
        </Button>
        <DeviceMenu items={deviceInfo} handleChange={handleMenuChange}/>
        <Button variant="contained" color="primary" onClick={run} sx={{height: '100%'}}>
          Run
        </Button>
        <Button variant="contained" color="secondary" sx={{height: '100%'}} onClick={async () => {
              await fetch(server('/reset'));
              window.localStorage.setItem("params", JSON.stringify({}));
              window.localStorage.setItem("title", '');
              window.location.reload();
            }
          }>
          Reset
        </Button>
        <FileNameInput value={title} setValue={val=>title=val}/>
        <Button variant="outlined" color="secondary" onMouseOver={()=>null} sx={{'border': 2, 'width': 2, maxHeight: 37, 'minWidth': 10, '&:hover': {'border': 2}}} onClick={() => {
              if (source === undefined && compareSource === undefined) {
                return console.log("No image to download.");
              }
              if (compare.length !== 0 && compareSource !== undefined) {
                download(source, title + '_compare.png', "image/png");
                download(compareSource, title + '.png', "image/png");
              } else {
                download(source, title + '.png', "image/png");
              }
            }
          }>
          <img alt="" src={downIcon} width="20" />
        </Button>
        <ToggleButtonGroup value={compare} color="primary" onChange={handleCompareToggle} sx={{width: 37, maxHeight: 37}} >
          <ToggleButton value={true} sx={{minHeight: 0, minWidth: 0, padding: "1px", '&.Mui-selected, &.Mui-selected:hover': {border: '2px solid #003262', backgroundColor: '#C4820E', color: '#C4820E'}}}>
            <img alt="" src={compIcon} width="30" />
          </ToggleButton>
        </ToggleButtonGroup>
        <ToggleButtonGroup disabled={!stereo} value={key} color="secondary" onChange={handleKeyToggle} sx={{width: 37, maxHeight: 37}} >
          <ToggleButton value={true} sx={{minHeight: 0, minWidth: 0, padding: "1px", '&.Mui-selected, &.Mui-selected:hover': {border: '2px solid #C4820E', color: '#003262'}}}>
            <img alt="" src={keyIcon} width="30" />
          </ToggleButton>
        </ToggleButtonGroup>
      </Stack>
    )
  }

  function ProgressMessageBox() {
    return (
      <Typography overflow={"auto"} component={'span'} marginLeft={5} style={{maxHeight: '14.5vh', width: '50vw', marginTop: 0}}>
        <pre style={{fontFamily: 'inherit', color: theme.palette.primary.main}}>
          {message}
        </pre>
      </Typography>
    )
  }

  function OutputUI() {
    return (
      <Stack marginTop={5} alignItems={"center"} style={{width: '100%'}} >
        <MainContent />
        <Divider orientation="horizontal" width='100%' sx={{border: 'thin solid #003262', marginTop: 3, marginBottom: 1}}/>
        <ButtonStack />
        <ProgressMessageBox />
      </Stack>
    )
  }

  return (
    <Grid container style={{display: "flex"}} direction="row" overflow={'clip'} >
      <Grid item style={{maxHeight: '100vh', overflow: 'hidden'}} borderRight='thin solid #003262' borderTop='thin solid #003262' width='20%'>
        {ParameterControl(stereo)}
      </Grid>
      <Grid item alignItems="right" id={"OutputGrid"} borderTop='thin solid #003262' borderRight='thin solid #003262' width='80%'>
        <OutputUI />
      </Grid>
    </Grid>
  )
}