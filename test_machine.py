from config import returnSharedCfg, returnMachineCfg
from machine import Machine
import os
import numpy as np
import asyncio
import sys
from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

def getAction(x):
    return np.ones((3, 2))*500*x + 500

s_cfg = returnSharedCfg()
m_cfg = returnMachineCfg()
machine = Machine(s_cfg, m_cfg)

for i in range(5):
    machine.sendAction(getAction(i))
    print(machine.getStates())
