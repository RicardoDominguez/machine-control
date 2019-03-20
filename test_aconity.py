from config import returnSharedCfg, returnMachineCfg
from machine import Machine
import os
import numpy as np
import asyncio

def getAction(i):
    return np.ones((3, 2))*i*500 + 400

s_cfg = returnSharedCfg()
m_cfg = returnMachineCfg()
machine = Machine(s_cfg, m_cfg)

for i in range(5):
    action = getAction(i)
    asyncio.run(machine.performLayer(action))
    print(machine.states)