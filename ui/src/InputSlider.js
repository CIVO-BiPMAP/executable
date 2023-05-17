import * as React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Slider from '@mui/material/Slider';
import MuiInput from '@mui/material/Input';
import Typography from "@mui/material/Typography";
import Checkbox from '@mui/material/Checkbox';
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";

const Input = styled(MuiInput)`
 width: 42px;
`;

function InputSlider(props) {
    const minimum = props.min ?? 0;
    const maximum = props.max ?? 100;
    const default_value = () => {
        //Check if the dictionary is not null, if the item is not null, but allow the item to be 0
        if (props.dict && (props.dict[props.id] || props.dict[props.id] === 0)) {
            return props.dict[props.id]
        }
        //if there is no loaded value, just return the default value
        return (props.default_value ?? (props.max / 2 ?? 50))
    }

    const [value, setValue] = React.useState(
        default_value(),
    );

    let init_state = props.default_enabled ?? false
    if ((props.input ?? "").toUpperCase() !== "CHECKBOX") {
      init_state = true
    }
    const [checked, setChecked] = React.useState(init_state);

    const handleCheckboxChange = () => {
        if (props.handleChange != null) {
            props.handleChange(props.id, value * (+!checked))
        }
        setChecked(!checked);
    };

    const handleSliderChange = (event, newValue) => {
        setValue(newValue);
        if (props.handleChange != null) {
            props.handleChange(props.id, newValue * (+checked))
        }
    };

    const handleInputChange = (event) => {
        setValue(event.target.value === '' ? '' : Number(event.target.value));
        if (props.handleChange != null) {
            props.handleChange(props.id, Number(event.target.value) * (+checked))
        }
    };

    // const handleBlur = () => {
    //     if (value < minimum) {
    //         setValue(minimum);
    //     } else if (value > maximum) {
    //         setValue(maximum);
    //     }
    // };

    // const scaleFunc = (x) => {return x};
    // if (props.scale != null) {
    //     const scaleFunc = (x) => {return Math.pow(10, x)};
    // }

    const InputOption = () => (
      <div>
        <Input
          value={value}
          size="small"
          onChange={handleInputChange}
          // onBlur={handleBlur}
          inputProps={{
            // step: (props.max - props.min)/100,
            min: (minimum),
            max: (maximum),
            type: 'number',
            'aria-labelledby': 'input-slider',
          }}
          style={{width: '100%'}}
          disabled={props.disable ?? false}
        />
      </div>
    );
    const SelectSecondary = () => {
        if ((props.input ?? "").toUpperCase() === "") {
            return (
              InputOption() //{'value': value, 'onChange': handleInputChange}
            )
        } else if (props.input.toUpperCase() === "CHECKBOX") {
            return (
                <Checkbox onChange={handleCheckboxChange} checked={checked} disabled={props.disable ?? false}/>
            )
        } else if (props.input.toUpperCase() === "NONE"){
            return null
        }
    }
    if (props.disable) {
        return (
            <ListItem
                key={props.id}
                disableGutters
            >
                <ListItemText primary={`${props.label}: ${value * (+checked)}`} />
            </ListItem>
        )
    }
    return (
        <ListItem
            key={props.id}
            disableGutters
        >
            <Box sx={{ width: 300}}>
                <Typography gutterBottom>
                    {props.label ?? ""}
                </Typography>
                <Grid container spacing={4} alignItems="center">
                    <Grid item xs>
                        <Slider
                            id={"main_slider"}
                            valueLabelDisplay={"auto"}
                            value={typeof value === 'number' ? value : 0}
                            onChange={handleSliderChange}
                            onChangeCommitted={handleSliderChange}
                            aria-labelledby="input-slider"
                            style={{marginLeft: '10px'}}
                            default_value={props.default_value ?? (props.max / 2 ?? 50)}
                            min={minimum}
                            max={maximum}
                            step={props.integer ? 1 : (maximum - minimum)/100}
                            // scale={scaleFunc}
                            marks={
                                [
                                    {value: minimum, label: minimum},
                                    {value: maximum, label: maximum}
                                ]
                            }
                            disabled={props.disable ?? false}
                        />
                    </Grid>
                    <Grid item marginLeft={1}>
                      {SelectSecondary()}
                    </Grid>
                </Grid>
            </Box>
        </ListItem>
    );
}

export default InputSlider;
