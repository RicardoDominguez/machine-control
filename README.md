# MACHINE Control

# Overall architecture

Software is divided into two sections:
 * Machine side - performs build and processes raw data
 * Cluster side - computes actions according to state of system

Communications is done through SFTP by passing files. The machine side establishes communication with the server, and takes care of setting and getting files.
 * The action array has a size of `N x 2`, where N is the number of parts being built.
 * The state array has a size of `N x 16`.

# Code structure

## Cluster side

Methods:
 * `getStates()` - reads current state of the system as communicated by the cluster side
 * `computeAction()` - returns control action given the current machine state
 * `sendAction()` - send the computed action to the machine side

Program flow:
 * `states` <- `getStates()`
 * `actions` <- `computeAction(states)`
 * `sendAction(action)`

## Machine side
 * `getActions()` - read actions communicated by the cluster side
 * `performLayer()` - starts the build of the next layer with the parameters specified
 * `getStates()` - reads the raw data outputted by the pyrometer while the layer is being built and processes it into states
 * `sendStates()` - send the state information to the cluster side

Program flow:
 * `actions` <- `getActions()`
 * `performLayer(actions)`
 * `states` <- `getStates()`
 * `sendStates(states)`

# AconitySTUDIO Python Client

## Create the client

```
from AconitySTUDIO_client import AconitySTUDIOPythonClient

login_data = {
    'rest_url' : 'http://192.168.1.1:9000',
    'ws_url' : 'ws://192.168.1.1:9000',
    'email' : 'admin@yourcompany.com',
    'password' : '<password>'
}

client = await AconitySTUDIOPythonClient.create(login_data)
```

## Configuring the client

Each job has a unique identifier which must be known in order to interact with said job. To automatically gather and set the correct job id for the Python Client use
```
 await client.get_job_id('TestJob')
```
This will automatically create an attribute ``job_id``. From now on, if any method of the Python Client would require a job id, you can omit this argument in the function call. If you chose to explicitly fill in this parameter in a function call, the clients own attribute (if it exists at all) will be ignored.


For normal operation of the Python Client, identifiers of the configuration
and the machine itself must be known aswell.
```
await client.get_machine_id('my_unique_machine_name')
await client.get_config_id('my_unique_config_name')
```

If multiple machines, configurations or jobs exist with the same name,
they need to be looked up in the browser url field and given to the Python Client manually:
```
client.job_id = '5c4bg4h21a00005a00581012'
client.machine_id = 'your_machine_id_gathered_from_browser_url_bar'
client.config_id = 'your_config_id_gathered_from_browser_url_bar'
```

## Job management

Includes starting, pausing, resuming and stopping.

For starting a job, we need to specify the job id, an execution script, and which layers shall be built with which parts. If we have set the attribute `job_id`
and all parts should be built, a job can be started like this:
```
layers = [1,3] #build layer 1,2,3

execution_script = \
'''layer = function(){
for(p:$p){
    $m.expose(p[next;$h],$c[scanner_1])
}
$m.add_layer($g)
}
repeat(layer)'''

await start_job(layers, execution_script)
```
This does not take care of starting a config or importing parameters from the config into a job. This needs to be done in the GUI beforehand. Of course, it is always possible to do the basic job configuration via the REST API in the Python Client, but no convenience functions exist to simplify these tasks.

After a job is paused (`await client.pause_job()`), one can change parameters (subpart *001_s1_vs* shall be exposed with a different Laserpower):
```
part_id = 1 #part_id of 001_s1_vs. See next section `Documentation of all functions`.
param = 'laser_power'
new_laser_power = 123
await client.change_part_parameter(part_id, param, new_value)
```

Changing a global parameter can be done via:
```
param = 'supply_factor'
new_value = 2.2
await client.change_global_parameter(param, new_value)
```

## Script execution
Use the `execute()` coroutine. For instance:
```
light_on = '$m.on($c[light])'
await client.execute(channel='manual', script=light_on)
movement = '$m.move_rel($c[slider], -180)'
await client.execute(channel='manual_move', script=movement)
```
These commands get executed on different channels. If a channel is occupied, any command sent to that channel will be ignored. The `execute` coroutine takes care of this because if you `await` it, it will only yield control to its caller once the channel is free again. This could be bypassed by commenting out some of the source code.