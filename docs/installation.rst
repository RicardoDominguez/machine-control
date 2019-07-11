============
Installing, running and enhancing the software
============

------------
Installing required dependencies
------------

The simplest way to install all required software packages is using `conda`.

The modeling and optimisation software requires TensorFlow. This package can
be run on CPU or GPU (if one is available), the latter offering up to 100x faster
run time performance.

For a CPU installation use::

  conda create -n tf-cpu python=3.5
  conda activate tf-cpu
  conda install tensorflow==1.10
  pip install dotmap scipy gpflow gym==0.9.4 pytest tqdm sklearn scikit-optimize

For a GPU installation use::

  conda create -n tf-gpu python=3.5
  conda activate tf-gpu
  conda install tensorflow-gpu==1.10
  pip install dotmap scipy gpflow gym==0.9.4 pytest tqdm sklearn scikit-optimize

If there are dependencies missing, these can be installed using `pip` or `conda`
in the typical Python fashion.

-----------
Running the software
-----------

First, one must set the desired configuration for the build. The configuration files are:

  - config_cluster.py
  - config_windows.py
  - config_dmbrl.py

Details regarding the available configurations can be found under the `Configuration` section.

After setting the desired configuration, one must:

  - Run `aconity.py` in the AconityComputer (i.e. using the Python IDLE), and wait until the command line displays `“Waiting for actions…”`
  - Open MobaXterm

    - Log into `USERNAME@scentrohpc.shef.ac.uk` and provide the pertinent password.
    - Run `source activate tf-cpu` (or whichever conda environment has the required dependencies)
    - Run `cd software-path` where `software-path` is the location of the software package on the remote server
    - Run `python cluster.py`
    - Wait until the command line displays “Waiting for states...”

  - Run `machine.py` (i.e. using the Python IDLE)

-----------
Enhancing the software
-----------

  - To implement a different control strategy, modify the function `computeAction()` in `cluster.py`.
  - To make changes to the current control strategy, modify the relevant files within `dmbrl/`
  - To change how the pyrometer measurements are converted into the low-dimensional features used for modeling and control, change the function `getStates()` from `machine.py`.

~~~~~~~~~~~~~~~~~~~~~~
Adding another sensor
~~~~~~~~~~~~~~~~~~~~~~

To add another sensor one could change the function `getStates()` from `machine.py` to resemble::

  states = np.zeros((n_parts, M+N)) # Initialise state vector
  for part in range(n_parts): # Read information for all parts being monitored
    # Load raw data from sensors
    data_sensor1 = loadSensor1(file_path_to_part_sensor1)
    data_sensor2 = loadSensor2(file_path_to_part_sensor2)

    # Process raw data from sensors
    state_sensor1 = processDataSensor1(data_sensor1) # vector with shape (N,)
    state_sensor2 = processDataSensor2(data_sensor2) # vector with shape (M,)

    # Combine
    state = np.concatenate((state_sensor1, state_sensor2)) # shape (N+M,)
    states[part] = state
  return states

One would also need to ensure that the new state representation is suitable for
modeling the system with sufficient accuracy. To do so, convert the build
data of interest into the state representation to be tested, train the model
with the given state representation and check its accuracy in making predictions
over previously unseen data (R2, RMSE...). Take the following script for reference::

  import numpy as np
  import tensorflow as tf
  from dotmap import DotMap
  import matplotlib.pyplot as plt
  from dmbrl.modeling.models import BNN
  from dmbrl.modeling.layers import FC

  states = np.load(states_file) # Dimension (n_samples, n_states)
  actions = np.load(actions_file) # Dimension (n_samples-1, n_actions)
  XU = np.concatenate((X[:-1], U), axis=1) # inputs to the model
  Yd = X[1:] # training targets

  # Split data into train and test sets
  test_ratio = 0.2
  num_test = int(X.shape[0] * test_ratio)
  permutation = np.random.permutation(X.shape[0])

  train_x, test_x = XU[permutation[num_test:]], XU[permutation[:num_test]]
  train_y, test_y = Yd[permutation[num_test:]], Yd[permutation[:num_test]]

  # Before this, define the model parameters model_in, model_out, n_layers, n_neurons, l_rate, wd_in, wd_hid, wd_out, num_networks
  sess = tf.Session()
  params = DotMap(name="model1", num_networks=num_networks, sess=sess)
  model = BNN(params)
  self.model.add(FC(n_neurons, input_dim=model_in, activation="swish", weight_decay=wd_in))
  for i in range(n_layers): self.model.add(FC(n_neurons, activation="swish", weight_decay=wd_hid))
  model.add(FC(model_out, weight_decay=wd_out))
  model.finalize(tf.train.AdamOptimizer, {"learning_rate": l_rate})

  # Train and make the predictions
  model.train(train_x, train_y, batch_size, training_epochs, rescale=True)
  predicted_y, var_y = model.predict(test_x) # Essentially the mean and variance of the prediction, as it is a probabilistic model

  # Compute metrics (R2 and RMSE should be previously define functions)
  r2_metric = R2(test_y, predicted_y)
  rmse_metric = RMSE(test_y, predicted_y)

When incorporating a new sensor, you most likely want to change how the scaling
of the data is done, so that the magnitude of all your state elements is similar.
Otherwise model performance will be degraded. To change the scaler functionality, make changes
to `dmbrl/modeling/utils/ModelScaler.py` as needed.

~~~~~~~~~~~~~~~~~~~~~~
Dividing a part into multiple "subparts"
~~~~~~~~~~~~~~~~~~~~~~

This approach aims to improve `intra`-layer temperature homogeneity. There are
two sides to this problem. First, one probably wants to model the entire part as
one. To do so, one should combine the sensory information obtained for all subparts
into a single state vector, in a similar manner to explanation above (but instead
of combining sensors, combining parts).

Secondly, you must configure the MPC class to output more parameters. For instance, if you have
7 sub-parts and want to select distinct laser power and scan configurations in each
of them, then your action vector should have dimension 7 * 2 = 14. To change the dimension of the
vector, simply ensure all `ac_lb`, `ac_up` and `constrains` variables fed to the `MPC` class
have the correct dimension, i.e check that ac_lb.shape==14, ac_up.shape==14 and same with `constrains`.

Then, you must make changes to the function `performLayer()` within `aconity.py` so that
the correct parts are addressed when changing the laser power and scan speed (must be able to handle
action inputs with secondary dimension > 2).
