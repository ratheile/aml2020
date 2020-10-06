
#%%
# Path hack.import sys, os
sys.path.insert(0, os.path.abspath('..'))
import project1_raffi.main as raffi
from modules import ConfigLoader

#%% some local testing:
cfg = ConfigLoader().from_file('base_cfg.yml')
print(raffi.run(cfg))