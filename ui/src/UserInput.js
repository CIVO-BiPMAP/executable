import * as React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import MuiInput from '@mui/material/Input';
import Typography from "@mui/material/Typography";
import {Button, Grid} from '@mui/material'

const Input = styled(MuiInput)`
 width: 42px;
`;

function UserInput(props) {
    const handleInputChange = (event) => {
        props.setText(event.target.value);
    };

    return (
        <Box marginBottom={0} >
            <Typography gutterBottom>
                {props.label || ""}
            </Typography>
            <Grid container spacing={1} alignItems='center' direction='row' >
                <Grid item>
                    <Input
                        value={props.text}
                        size="small"
                        onChange={handleInputChange}
                        style={{minWidth: 100}}
                        disabled={props.disable || false}
                        />
                </Grid>
                <Grid item>
                    <Button variant="contained" onClick={props.buttonHandler} disabled={props.disable || false} marginleft={15}>
                        submit
                    </Button>
                </Grid>
            </Grid>
        </Box>
    );
}

export default UserInput;
