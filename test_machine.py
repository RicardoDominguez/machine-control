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
machine = Machine(s_cfg, m_cfg)
machine.loop()
