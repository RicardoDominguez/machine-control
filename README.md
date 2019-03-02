# MACHINE Control

# Overall architecture

Software is divided into two sections:
 * Machine side - performs build and processes raw data
 * Cluster side - computes actions according to state of system

Communications is done through SFTP by passing files. The machine side establishes communication with the server, and takes care of setting and getting files.
 * The action array has a size of `N x 2`, where N is the number of parts being built.
 * The state array has a size of `N x 16`.


# Cluster side

Methods:
 * `getStates()` - reads current state of the system as communicated by the cluster side
 * `computeAction()` - returns control action given the current machine state
 * `sendAction()` - send the computed action to the machine side

Program flow:
 * `states` <- `getStates()`
 * `actions` <- `computeAction(states)`
 * `sendAction(action)`

# Machine side
 * `getActions()` - read actions communicated by the cluster side
 * `performLayer()` - starts the build of the next layer with the parameters specified
 * `getStates()` - reads the raw data outputted by the pyrometer while the layer is being built and processes it into states
 * `sendStates()` - send the state information to the cluster side

Program flow:
 * `actions` <- `getActions()`
 * `performLayer(actions)`
 * `states` <- `getStates()`
 * `sendStates(states)`
