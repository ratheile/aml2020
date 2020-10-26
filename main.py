from modules import ConfigLoader

import coloredlogs, logging
coloredlogs.install()

import argparse
import os
import glob

from enum import Enum

from project1_raffi import main as raffi
from project2_ines import main as ines
from project1_ffu import main as ffu
from project1 import main as project1
from project1.estimator import Project1Estimator

class User(Enum):
  ines = 'ines'
  ffu = 'ffu'
  raffi = 'raffi'
  grid = 'grid'

  def __str__(self):
      return self.value

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":

  # Parse arguments
  parser = argparse.ArgumentParser()

  input_grp = parser.add_mutually_exclusive_group(required=True)

  parser.add_argument('--env', type=file_path, default='env/env.yml',
    help='The environment yaml file.')

  parser.add_argument('--slice', type=file_path, help='The slice for user grid')

  input_grp.add_argument('--cfg', type=file_path,
    help='The main config yaml file.')

  input_grp.add_argument('--dir', type=dir_path,
    help='The main config directory.')

  parser.add_argument('--user', type=User, choices=list(User))

  args = parser.parse_args()
  env_cfg_path = args.env
  user: User = args.user
  run_cfg_dir = args.dir
  slice_path = args.slice

  # Load configs
  env_cfg = ConfigLoader().from_file(env_cfg_path)

  if args.cfg:
    logging.info('single cfg mode')
    run_cfg_paths = [args.cfg]
  elif args.dir:
    logging.info("directory cfg mode")
    run_cfg_paths = glob.glob(f'{run_cfg_dir}/*.yml')

  logging.info(f'Loading the data from: {env_cfg["datasets/project2/path"]}')

  if user is User.raffi:
    # my "main" function
    for id_ex, run_cfg_path in enumerate(run_cfg_paths):
      name = os.path.basename(run_cfg_path)
      run_cfg = ConfigLoader().from_file(run_cfg_path)
      logging.info(f'running experiment {id_ex + 1} with name {name}')
      raffi.run(run_cfg, env_cfg)

  elif user is User.ffu:
    for id_ex, run_cfg_path in enumerate(run_cfg_paths):
      name = os.path.basename(run_cfg_path)
      run_cfg = ConfigLoader().from_file(run_cfg_path)
      logging.info(f'running experiment {id_ex + 1} with name {name}')
      ffu.run(run_cfg, env_cfg)
    
  elif user is User.ines:
    for id_ex, run_cfg_path in enumerate(run_cfg_paths):
      name = os.path.basename(run_cfg_path)
      run_cfg = ConfigLoader().from_file(run_cfg_path)
      logging.info(f'running experiment {id_ex + 1} with name {name}')
      ines.run(run_cfg, env_cfg) # this is the run function from you project-level main.py
  
  elif user is User.grid:
    # Gridsearch Impl 
    for id_ex, run_cfg_path in enumerate(run_cfg_paths):
      name = os.path.basename(run_cfg_path)
      run_cfg = ConfigLoader().from_file(run_cfg_path)
      slice_cfg = ConfigLoader().from_file(slice_path)
      logging.info(f'running experiment {id_ex + 1} with name {name}')
      project1.gridsearch(run_cfg, env_cfg, slice_cfg) 

  else:
    # no user
    for id_ex, run_cfg_path in enumerate(run_cfg_paths):
      name = os.path.basename(run_cfg_path)
      run_cfg = ConfigLoader().from_file(run_cfg_path)
      logging.info(f'running experiment {id_ex + 1} with name {name}')
      project1.run(run_cfg, env_cfg) # this is the run function from you project-level main.py