from config import returnSharedCfg
from cluster import Cluster

s_cfg = returnSharedCfg()
cluster = Cluster(s_cfg)
cluster.loop()