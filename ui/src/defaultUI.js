import './app.css';
import InputSlider from './InputSlider';
import * as React from 'react';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Grid from "@mui/material/Grid";
import List from "@mui/material/List";
import InputSelect from "./InputSelect";
import InputCheck from "./InputCheck";
import { handleParameterChanges, getParameterDict, Container } from './app';

function StimulusPars(props) {
  return (
    <Grid container spacing={0} direction="column">
      <InputSlider class="param" id={"vx"} label={"X Velocity (cm/s)"} default_value={10} max={50} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"objSize"} label={"Object Size (cm)"} default_value={0.05} max={0.3} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"recordingLength"} label={"Recording Length (s)"} default_value={0.5} max={2} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
    </Grid>
  );
}

function ViewingPars(props) {
  return (
    <Grid container spacing={0} direction="column">
      <InputSlider class="param" id={"viewing_D"} label={"Viewing Distance (cm)"} default_value={50} max={100} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputCheck class="param" id={"ve"} label={"Object Tracking"} default_value={true} disable={props.disable} trueWord="ON" falseWord="OFF" handleChange={handleParameterChanges} dict={getParameterDict()}/>
    </Grid>
  );
}
function contrastSelector(props) {
  if (getParameterDict()["RGBmode"] && getParameterDict()["RGBmode"] !== 'bw') {
    return (
      <div>
        <InputSlider class="param" id={"contrast_R"} label={"Contrast R"} default_value={1} max={1} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
        <InputSlider class="param" id={"contrast_G"} label={"Contrast G"} default_value={1} max={1} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
        <InputSlider class="param" id={"contrast_B"} label={"Contrast B"} default_value={1} max={1} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      </div>
    )
  } else {
    return (
      <InputSlider class="param" id={"contrast"} label={"Contrast"} default_value={1} max={1} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
    )
  }
  }
  
function DisplayPars(props) {
  return (
    <Grid container spacing={0} direction="column">
      <InputSlider class="param" id={"nFlash"} label={"Number of Flashes"} default_value={1} min={1} max={3} input={"none"} integer disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSelect class="param" id={"RGBmode"} label={"RGB Mode"} default_value='bw' options={['bw', 'seq', 'simul']} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputCheck class="param" id={"spatialOffset"} label={"Spatial Offset"} default_value={false} disable={props.disable} trueWord="ON" falseWord="OFF" handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"frmRate"} label={"Capture Rate (Hz)"} default_value={60} max={360} integer disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"holdInterval"} label={"Hold Interval"} default_value={1} max={1} input={"none"} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"pxlResponseT"} label={"Pixel Response (ms)"} default_value={0} max={10} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"dpi"} label={"DPI"} default_value={254} max={1000} disable={props.disable} integer handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"fillF"} label={"Fill Factor"} default_value={1} max={1} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      <InputSlider class="param" id={"luminance"} label={"Luminance (cd/m^2)"} default_value={100} max={1000} disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
      {contrastSelector(props)}
      <InputSlider class="param" id={"antialiasingF"} label={"Antialiasing (Gaussian)"} default_value={3} max={10} integer input={"checkbox"} default_enabled={false} disable={props.disable} handleChange={handleParameterChanges}/>
    </Grid>
  );
}
  
function ParameterControl(props) {
  return (
    <Container>
      <StimulusPars disable={props.value !== '0'} />
      <DisplayPars disable={props.value !== '1'} />
      <ViewingPars disable={props.value !== '2'} />
    </Container>
  );
}

function DefaultUI() {
  const [value, setValue] = React.useState(window.localStorage.getItem('tabState') ?? '0');
  const handleChange = (event, newValue) => { 
    setValue(newValue);
    window.localStorage.setItem('tabState', newValue);

  };
  return (
    <div className="side_bar" id={"side_tabs"} align={"down"}>
      <Tabs
        value={value}
        onChange={handleChange}
        textColor="primary"
        indicatorColor="primary"
        aria-label="primary tabs example"
        style={{borderBottom: 'thin solid #003262'}}
        >
        <Tab value='0' label="Stimulus" style={{width: '33%'}}/>
        <Tab value='1' label="Display" style={{width: '33%'}}/>
        <Tab value='2' label="Viewing" style={{width: '33%'}}/>
      </Tabs>
      <List sx={{ width: '100%', maxHeight: '90%', overflow: 'auto', bgcolor: 'background.paper'}}>
        <ParameterControl value={value}/>
      </List>
    </div>
  );
};

export default DefaultUI;
