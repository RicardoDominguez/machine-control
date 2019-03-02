from config import returnSharedCfg, returnMachineCfg
from machine import Machine
import os
import numpy as np

def getStates(i):
    return np.ones((100, 16))*i

s_cfg = returnSharedCfg()
m_cfg = returnMachineCfg()
machine = Machine(s_cfg, m_cfg)

for i in range(5):
    print(machine.getActions())
    machine.sendStates(getStates(i))