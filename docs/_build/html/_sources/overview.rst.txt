========
Overview
========

This software package offers real-time modeling and optimisation of a laser
powder bed fusion process (AconityMINI). To do so, three scripts are executed
simultenously: `aconity.py`, `machine.py` and `cluster.py`, the two former
executed locally in the AconityMINI computer and the latter executed in a remote
server for enhanced run-time performance.

  - `aconity.py`: Makes use of the API provided by Aconity to automatically start,
    pause and resume a build, and to change individual part parameters in real-time.
  - `machine.py`: Reads the raw sensory data outputted by the aconity machine,
    processes it into a low-dimensional state vector and uploads it a remote server for
    parameter optimisation.
  - `cluster.py`: Computes optimal process parameters, at each layer, given
    feedback obtained from the machine sensors. Based on the deep reinforcement
    learning algorithm Probability Ensembles with Trajectory Sampling.

------------
Program flow
------------

  - Layer is started by `performLayer()` in `aconity.py`
  - Pyrometer data is read and processed in real-time by `getStates()` in `machine.py`
  - When the layer is completed and all data has been read, the low-dimensional
    processed states are sent to the remote server by `sendStates()` in `machine.py`
  - The states are received at the remote server by `getStates()` in `cluster.py`
  - A new control action is computed (build parameters are optimised) according
    to the received feedback by `computeAction` in `cluster.py`
  - The computed actions are saved to the remote server by `sendAction()` in `cluster.py`
  - The computed actions are downloaded locally by `getActions()` in `machine.py`
  - A new layer is built using the updated parameters by `performLayer()` in `aconity.py`

The Aconity API software package provided by Aconity3D must be installed in the
computer connected to the Aconity machine according to Aconity's guidelines. The
two files containing the bulk of the functionality of the API are `AconitySTUDIO_client.py`
and `AconitySTUDIO_utils.py`.
