from config_cluster import returnSharedCfg, returnClusterPretrainedCfg
from cluster import Cluster
import numpy as np
import tensorflow as tf

s_cfg = returnSharedCfg()
c_cfg = returnClusterPretrainedCfg()

cluster = Cluster(s_cfg, c_cfg)
cluster.loop()
