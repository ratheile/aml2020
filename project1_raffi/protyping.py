
#%%
# Path hack.import sys, os
sys.path.insert(0, os.path.abspath('..'))
import project1_raffi.main as raffi
from modules import ConfigLoader

#%% some local testing:
run_cfg = ConfigLoader().from_file('base_cfg.yml')
env_cfg = ConfigLoader().from_file('../env/env.yml')
print(raffi.run(run_cfg, env_cfg))
