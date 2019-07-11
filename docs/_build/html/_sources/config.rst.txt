========================
Configuration parameters
========================

The desired configuration for the build is set on the following files:

  - `config_windows.py`: General configuration parameters concerning the build, such as number of parts, build parameters, ...
  - `config_dmbrl.py`: Low-level control specific configuration. Generally one would not need to change this file, but rather `config_cluster.py`
  - `config_cluster.py`: Control configuration, divided into 'pretrained' (model trained using data collected previously) and `unfamiliar` (model learned in real-time).

----------------------
config_windows.py
----------------------

  - LASER_ON (bool): Laser is enabled when True.
  - JOB_NAME (str): Job name as displayed in the AconitySTUDIO web application.
  - LAYERS (array of int): Layer range to be built, as [layer_min, layer_max].
  - N_PARTS (int): Number of parts to be built (not regarding ignored parts).
  - N_STATES (int): Number of low-dimensional states used for the processing of the raw pyrometer data.
  - TEMPERATURE_TARGET (float): Temperature target in mV.
  - N_PARTS_IGNORED (int): Number of additional parts to be built on top of `N_PARTS` (pyrometer may not record data for the first few parts).
  - IGNORED_PARTS_SPEED (float): Scan speed used for parts being "ignored".
  - IGNORED_PARTS_POWER (float): Laser power used for parts being "ignored".
  - N_PARTS_FIXED_PARAMS (int): Number of parts built using fixed build parameters.
  - FIXED_PARAMS (array): Parameters to be used for those parts being built with fixed build parameters, as [speed (m/s), power (W)]
  - SLEEP_TIME_READING_FILES (float): Time between a sensor data file being first detected and attempting to read it. Prevents errors emerging from opening the file while it is still being written.
  - PART_DELTA (int): Parts of interest may increase 1 by 1, or 3 by 3 (refer to the AconitySTUDIO web application).

--------------------------------------------
config_dmbrl.py and config_cluster.py
--------------------------------------------

  - ctrl_cfg: Configuration parameters for the control algorithm.

      -dO: dimensionality of observations
      -dU: dimensionality of control inputs
      - per: How often the action sequence will be optimized, i.e, for per=1 it is reoptimized at every call to `MPC.act()`.
      - constrains: [[np.array([min v, min q]), np.array([max v, max q])], [min q/v, max q/v], [min q/sqrt(v), max q/sqrt(v)]]
      - prop_cfg: Configuration parameters for modeling and uncertainty propagation.

          - model_pretrained: `True` if model used for MPC has been trained on previous data, `False` otherwise.
          - model_init_cfg: Configuration parameters for model initialisation.

              - ensemble_size: Number of models within the ensemble.
              - load_model: `True` for a pretrained model to be loaded upon initialisation.
              - model_dir: Directory in which the model files (.mat, .nns) are located.
              - model_name: Name of the model files (model_dir/model_name.mat or model_dir/model_name.nns)

          - model_train_cfg: Configuration parameters for model training optimisation

              - batch_size: Batch size.
              - epochs: Number of training epochs.
              - hide_progress: If 'True', additional information regarding model training is printed.

          - npart: Number of particles used for uncertainty propagation.
          - model_in: Number of inputs to the model.
          - model_out: Number of outputs to the model.
          - n_layers: Number of hidden layers.
          - n_neurons: Number of neurons per hidden layer.
          - learning_rate: Learning rate.
          - wd_in: Weight decay for the input layer neurons.
          - wd_hid: Weight decay for the hidden layer neurons.
          - wd_out: Weight decay for the output layer neurons.

      - opt_cfg: Configuration parameters for optimisation.

          - mode: Uncertainty propagation method.
          - plan_hor: Planning horizon for the model predictive control algorithm.
          - cfg

              - popsize: Number of cost evaluations per iteration.
              - max_iters: Maximum number of optimisation iterations.
              - num_elites: Number of elites.
              - alpha: Alpha parametero of the CEM optimisation algorithm.
              - eps: Epsilon parameter of the CEM optimisation algorithm.

          - prop_cfg

              - mode: Uncertainty propagation method, ie "TSinf"

      - change_target: True if multiple setpoints used, i.e. 980 and 1010
      - n_parts_targets: Number of parts to be built for each target
      - targets: Different temperature setpoints to be used (must be of same length as `n_parts_targets`)
      - force: Configuration parameters to periodically overwrite ("force") predefined build parameters

          - on: Force functionality enabled if True
          - start_part: First part where functionality is enabled (disregarding the first few ignored parts)
          - n_parts: Number of parts for which the functionality is enabled
          - n_repeats: Number of consecutive layers for which inputs are forced. For [1,2], n_parts will be forced only once (periodically), while a further n_parts will be forced two times consecutively (periodically)
          - init_buffer: Initial number of layers for which parameters are not forced
          - upper_init: Upper bound is initialised to this.
          - upper_delta: Upper bound increases by this. For instance, for upper_init=105 and upper_delta=5, the upper bound sequence will be 105, 110, 115...
          - lower_init: Lower bound is initialised to this.
          - lower_delta: Lower bound is increased by this. For instance, for lower_init=65 and lower_delta=-5, the lower bound sequence will be 60, 55, 50...
          - fixed_speed: For the forced parameters, power will be adjusted but mark speed will be kept fixed to this value.
