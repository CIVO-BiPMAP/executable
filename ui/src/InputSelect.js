import * as React from 'react';
import Box from '@mui/material/Box';
import Typography from "@mui/material/Typography";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import {MenuItem, Select} from "@mui/material";

function InputSelect(props) {
    const default_value = () => {
        //Check if the dictionary is not null, if the item is not null, but allow the item to be 0
        if (props.dict && (props.dict[props.id] || props.dict[props.id] === 0)) {
            return props.dict[props.id]
        }
        //if there is no loaded value, just return the default value
        return (props.default_value || '')
    }

    const [value, setValue] = React.useState(
        default_value().toString(),
    );

    const handleChange = (event) => {
      let newVal = (event.target.value || '').toString()
      setValue(newVal);
      if (props.handleChange != null) {
        props.handleChange(props.id, newVal)
      }
    };

    if (props.disable) {
      return (
        <ListItem
          key={props.id}
          disableGutters
        >
          <ListItemText sx={{color: 'primary'}} primary={`${props.label}: ${value}`} />
        </ListItem>
      )
    }

    return (
      <ListItem
        key={props.id}
        disableGutters
      >
        <Box sx={{ width: 300, display: 'flex'}}>
          <Typography gutterBottom>
            {props.label || ""}
          </Typography>
          <Select
            // labelId="demo-simple-select-label"
            id={`select_${props.id}`}
            value={value}
            // label="Age"
            sx={{minWidth: 150, marginLeft: 'auto', marginRight: '8'}}
            onChange={handleChange}
          >
            {props.options.map((option) => {
              return <MenuItem key={option} value={option}>{option}</MenuItem>;
            })}
          </Select>
        </Box>
      </ListItem>
    );
}

export default InputSelect;
