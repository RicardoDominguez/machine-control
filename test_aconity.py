from config import returnSharedCfg, returnMachineCfg
from aconity import Aconity
import os
import numpy as np
import asyncio
import sys
from AconitySTUDIO_client import AconitySTUDIOPythonClient
from AconitySTUDIO_client import utils

async def main():
    s_cfg = returnSharedCfg()
    m_cfg = returnMachineCfg()
    aconity = Aconity(s_cfg, m_cfg)

    utils.log_setup(sys.argv[0], directory_path='')
    await aconity.initAconity()
    await aconity.loop()

asyncio.run(main(), debug=True)