import './app.css';
import Typography from '@mui/material/Typography';
import StereoUI from "./stereoUI";
import DefaultUI from "./defaultUI";
import {server} from './index';
import $ from 'jquery';

let parameter_dict = {};

const saveParameters = () => {
  window.localStorage.setItem("params", JSON.stringify(parameter_dict));
  console.log(parameter_dict)
}

export const handleParameterChanges = (label, newValue) => {
  console.log(newValue)
  parameter_dict[label] = newValue
  saveParameters()
  console.log(parameter_dict)
};

export const getParameterDict = () => parameter_dict;

const loadParameters = () => {
  console.log(parameter_dict)
  parameter_dict = JSON.parse(window.localStorage.getItem("params")) || {};
  console.log("LOADED")
  console.log(parameter_dict)
}

loadParameters()

export function Container(props) {
  return (
    <Typography component="div" style={{ padding: 8 * 3}} align={'left'}>
      {props.children}
    </Typography>
  );
}

export function Conditional(props) {
  if (props.condition) {
      return props.item;  
  }
  return <div />;
}

function getDocumentationString() {
  const documentation = require('./assets/documentation.json');

  const analyzeDocumentation = (ind) => {
    let doc = documentation[ind];
    return `UI Version ${ind} released on ${doc.date}` + '\nRelease Notes:\n' + doc.notes + '\n\n'; //eslint-disable-line
  }

  return Object.keys(documentation).reverse().map((k) => analyzeDocumentation(k)).join('');
}

export const documentationString = getDocumentationString()

export function ParameterControl(is_stereo) {
  if (is_stereo) {
    return <StereoUI />
  }
  return <DefaultUI />
}

export function handleMenuChange(item) {
  $.post(server('/setdevice'), {
    'device': item
  });
  window.localStorage.setItem("device", item)
}

export function getStereo() {
  return window.localStorage.getItem("stereo") === "true";
}