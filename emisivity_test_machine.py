from config_windows import returnSharedCfg, returnMachineCfg
from machine import Machine
import os
import numpy as np
import asyncio
import sys
from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

s_cfg = returnSharedCfg()
m_cfg = returnMachineCfg()
m_cfg.aconity.open_loop=np.empty((0,2))
machine = Machine(s_cfg, m_cfg)
machine.test_loop()
