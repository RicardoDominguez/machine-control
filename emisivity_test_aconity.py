from config_windows import returnSharedCfg, returnMachineCfg
from aconity import Aconity
import os
import numpy as np
import asyncio
import sys
from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

# Please configure actions according to the number of parts
actions = np.array([[1, 75],
                   [1, 140],
                   [0.57, 110],
                   [1.8, 110]])

async def main():
    s_cfg = returnSharedCfg()
    m_cfg = returnMachineCfg()
    m_cfg.aconity.open_loop=np.empty((0,2))
    aconity = Aconity(s_cfg, m_cfg)

    utils.log_setup(sys.argv[0], directory_path='')
    await aconity.initAconity()
    await aconity.test_loop(actions)

asyncio.run(main(), debug=True)
