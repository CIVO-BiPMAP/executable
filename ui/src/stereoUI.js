import './app.css';
import InputSlider from './InputSlider';
import * as React from 'react';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Grid from "@mui/material/Grid";
import List from "@mui/material/List";
import InputSelect from "./InputSelect";
import InputCheck from "./InputCheck";
import { handleParameterChanges, getParameterDict, Container, Conditional } from './app';

function StereoUI() {
    const [value, setValue] = React.useState('0');
    const handleChange = (event, newValue) => { setValue(newValue); };
    const [isFlash, setIsFlash] = React.useState((getParameterDict()["capture"] !== "alt" && getParameterDict()["capture"]) ?? false);

    function handleStereoParameterChanges(label, newValue) {
        handleParameterChanges(label, newValue);
        if (label === 'capture') {
            setIsFlash(newValue !== 'alt');
        }
        console.log(isFlash)
    }
    
    function Parameters(props) {
        return (
            <Grid container spacing={0} direction="column">
                <InputSelect class="param" id={"capture"} label={"Capture Mode"} default_value='alt' options={['sim', 'alt']} disable={props.disable} handleChange={handleStereoParameterChanges} dict={getParameterDict()}/>
                <InputSelect class="param" id={"_pmode"} label={"Presentation Mode"} default_value='alt' options={['alt']} disable={props.disable} handleChange={handleStereoParameterChanges} dict={getParameterDict()}/>
                <InputSlider class="param" id={"vx"} label={"X Velocity (cm/s)"} default_value={10} max={50} disable={props.disable} handleChange={handleStereoParameterChanges} dict={getParameterDict()}/>
                <InputSlider class="param" id={"viewing_D"} label={"Viewing Distance (cm)"} default_value={50} max={100} disable={props.disable} handleChange={handleStereoParameterChanges} dict={getParameterDict()}/>
                <InputCheck class="param" id={"ve"} label={"Object Tracking"} default_value={true} disable={props.disable} trueWord="ON" falseWord="OFF" handleChange={handleStereoParameterChanges} dict={getParameterDict()}/>
                <InputSlider class="param" id={"frmRate"} label={"Capture Rate (Hz)"} default_value={60} max={360} integer disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
                <InputSlider class="param" id={"holdInterval"} label={"Hold Interval"} default_value={1} max={1} input={"none"} disable={props.disable} handleChange={handleStereoParameterChanges} dict={getParameterDict()}/>
                <Conditional condition={isFlash} item={(
                    <InputSlider class="param" id={"nFlash"} label={"Number of Flashes"} default_value={1} min={1} max={3} integer disable={props.disable} handleChange={handleParameterChanges} dict={getParameterDict()}/>
                    )} />
            </Grid>
        );
    }

    return (
        <div className="side_bar" id={"side_tabs"} align={"down"}>
            <Tabs
                value={value}
                onChange={handleChange}
                textColor="primary"
                indicatorColor="primary"
                aria-label="primary tabs example"
                style={{borderBottom: 'thin solid #003262', width: '100%'}}
                >
                <Tab value='0' label="Stereo" style={{width: '100%'}}/>
            </Tabs>
            <List sx={{ width: '100%', maxHeight: '90%', overflow: 'auto', bgcolor: 'background.paper'}}>
                <Container>
                    <Parameters/>
                </Container>
            </List>
        </div>
    );
}

export default StereoUI;