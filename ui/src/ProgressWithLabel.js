import * as React from 'react';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import ucb from "./assets/ucb_logo_yellow.png";


export function LinearProgressWithLabel(props) {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center' }} >
            <Box >
                <LinearProgress sx={{'& .MuiProgress-thumb': {
                        borderRadius: '100px',
                    }}} variant="determinate" {...props}/>
            </Box>
            <Box marginLeft={1} sx={{ minWidth: 40 }}>
                <Typography variant="body2" color="text.secondary">{`${Math.round(
                    props.value,
                )}%`}</Typography>
            </Box>
        </Box>
    );
};

export function CircularProgressWithLabel(props) {
    return (
        <Box sx={{ margin: 'auto', height: '70.5vh', position: 'relative'}}>
            <img src={ucb} alt={"failed to load"} width={200} style={{margin: 'auto', marginTop: '25vh'}}/>
            <CircularProgress
                variant='determinate'
                thickness={1.3}
                size={212}
                value={props.value}
                sx={{
                position: 'absolute',
                top: -6,
                left: -6,
                zIndex: 1,
                margin: 0,
                marginTop: '25vh'
                }}
            />
            <Box sx={{width: '100%', margin: 'auto'}}>
                <Typography variant="body2" color="primary" fontWeight='bold' margin='auto' textAlign={'center'} fontSize={17}>{`${Math.round(
                    props.value,
                )}%`}</Typography>
            </Box>
        </Box>
    );
};
