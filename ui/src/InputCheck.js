import * as React from 'react';
import Box from '@mui/material/Box';
import Typography from "@mui/material/Typography";
import Checkbox from '@mui/material/Checkbox';
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";

function InputCheck(props) {
    const default_value = () => {
        //Check if the dictionary is not null, if the item is not null, but allow the item to be 0
        if (props.dict && (props.dict[props.id] || props.dict[props.id] === false)) {
            return props.dict[props.id]
        }
        //if there is no loaded value, just return the default value
        return (props.default_value || 0)
    }

    const falseWord = (props.falseWord || "false")
    const trueWord = (props.trueWord || "true")

    const [value, setValue] = React.useState(
        Boolean(default_value()),
    );

    const handleChange = () => {
      if (props.handleChange != null) {
        props.handleChange(props.id, !value)
      }
      setValue(!value);
    };

    if (props.disable) {
      return (
        <ListItem
          key={props.id}
          disableGutters
        >
          <ListItemText primary={`${props.label}: ${value ? trueWord : falseWord}`} />
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
          <Checkbox
            sx={{minWidth: 150, marginLeft: 'auto', marginRight: '8'}}
            onChange={handleChange}
            checked={value}
            disabled={props.disable || false}
          />
        </Box>
      </ListItem>
    );
}

export default InputCheck;
