import './app.css';
import Logo from './assets/ucb_logo_yellow.png';
import {React, useEffect, useState} from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import './loading.css';
import Content from './pythonConnector';
import {createTheme, ThemeProvider} from '@mui/material';

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="logo-container">
        <div className="logo-spinner">
          <img src={Logo} alt="Logo" className="logo" width="200"/>
          <div className="loading-spinner"></div>
        </div>
      </div>
      <h1 className="loading">BiPMAP</h1>
      <p className="loading">We're getting things ready for you...</p>
    </div>
  );
}


export function server(url) {
  return 'http://localhost:3334' + url
}

export const theme = createTheme({
  palette: {
    primary: {
      main: '#003262'
    },
    secondary: {
      main: '#C4820E'
    }
  }
});

if (!window.localStorage.getItem("params")) {
  window.localStorage.setItem("params", JSON.stringify({}));
}

function ContentLoader() {
  const [contentReady, setContentReady] = useState(false);

  async function checkServer() {
    try {
      const response = await fetch(server('/getinfo'));
      const data = await response.json();
      if (data.device_info) {
        setContentReady(true);
        return
      }
    } catch (error) {
      console.error(error);
    }
    setTimeout(checkServer, 500)
  }

  useEffect(() => {checkServer()}, []);

  if (!contentReady) {
    return <LoadingScreen />;
  }
  return <Content />;
}

document.title = "BiPMAP";
ReactDOM.render(
  <ThemeProvider theme={theme}>
    <ContentLoader />
  </ThemeProvider>,
  document.getElementById('root')
);
