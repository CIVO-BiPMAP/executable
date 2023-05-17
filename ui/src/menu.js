import React from "react";
import {Menu, MenuItem, Button} from "@mui/material";

function MenuItems(props) {
  return props.items.map((item, index) => (
    <MenuItem key={index} onClick={() => props.onClick(index)}>{item}</MenuItem>
    )
  );
}

export default function DeviceMenu(props) {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [label, setLabel] = React.useState(props.items[window.localStorage.getItem("device")] || 'Device');
  const open = Boolean(anchorEl);

  window.addEventListener('load', () => {
    if (props.handleChange) {
      props.handleChange(window.localStorage.getItem("device"));
    }
  });

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget); 
  };

  const handleClose = (item) => {
    setAnchorEl(null);
    if (item >= 0 && item < props.items.length) {
      setLabel(props.items[item]);
      if (props.handleChange) {
        props.handleChange(item);
      }
    }
  };

  return (
    <div>
      <Button
        variant="outlined"
        color="secondary"
        id="demo-positioned-button"
        aria-controls={open ? 'demo-positioned-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
        >
        {label}
      </Button>
      <Menu
        id="demo-positioned-menu"
        aria-labelledby="demo-positioned-button"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
      >
        <MenuItems items={props.items} onClick={handleClose} />
      </Menu>
    </div>
  );
}