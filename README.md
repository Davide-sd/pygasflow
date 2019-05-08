## pygasflow

pygasflow provides a few handful functions to quickly perform quasi-1D ideal gasdynamic computations with Python (see requirements below).

At the moment, the following flow relations are implemented:
* Isentropic flow
* Fanno flow
* Rayleigh flow
* Shock wave relations (normal shock wave and oblique shock wave)

This repository is still a Work In Progress and need to be properly tested. Use it at your own risk. If you find any errors, submit an issue or a pull request!

## Requirements

* Python >= 3.6
* numpy
* scipy

## Usage

The code is well documented. I went for a natural language nomenclature since (I bet) most of us are using and advanced editor with autocompletion. For instance, the critical temperature T/T* is defined as Critical_Temperature_Ratio across the different flows, and so on.