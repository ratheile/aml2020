from modules import ConfigLoader

import coloredlogs, logging
coloredlogs.install()

import argparse
import os
import glob

from enum import Enum

from project1_raffi import main as raffi

from project1_ines import main as ines
# from project2_ines import main as ines
from project1_ffu import main as ffu
from project1 import main as project1
from project2 import main as project2
from project3 import main as project3
from project3.estimator import Project3Estimator

class User(Enum):
  ines = 'ines'
  ffu = 'ffu'
  raffi = 'raffi'
  default = 'default'

class Type(Enum):
  run = 'run'
  grid = 'grid'  # grid search
  cv = 'cv'  # cross validation
  convert = 'convert' # convert data to joblib

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

  parser.add_argument('--slice', type=file_path, help='The slice yaml file for grid search')

  input_grp.add_argument('--cfg', type=file_path,
    help='The main config yaml file.')

  input_grp.add_argument('--dir', type=dir_path,
    help='The main config directory.')

  parser.add_argument('--user', type=User, choices=list(User), default=User.default)
  parser.add_argument('--type', type=Type, choices=list(Type), default=Type.run)

  args = parser.parse_args()
  env_cfg_path = args.env
  run_cfg_dir = args.dir
  slice_path = args.slice

  user: User = args.user
  etype: Type = args.type

  # Load configs
  env_cfg = ConfigLoader().from_file(env_cfg_path)

  if args.cfg:
    logging.info('single cfg mode')
    run_cfg_paths = [args.cfg]
  elif args.dir:
    logging.info("directory cfg mode")
    run_cfg_paths = glob.glob(f'{run_cfg_dir}/*.yml')

  logging.info(f'Loading the data from: {env_cfg["datasets/project3/path"]}')

  # Modes available for a given project
  # e = env_cfg r = run_cfg
  project3_catalog = {
    User.ffu : {
      Type.run: lambda r,e: ffu.run(r,e)
    },
    User.ines : {
      Type.run: lambda r,e: ines.run(r,e)
    },
    User.raffi: { },
    User.default: {
      Type.run: lambda r,e: project3.run(r,e), 
      Type.cv: lambda r,e: project3.cross_validate(r,e),
      Type.grid: lambda r,e,s: project3.gridsearch(r,e,s),
      Type.convert: lambda r,e: project3.convert_data(r,e) 
    },
  }

  #  Checks if input is valid
  pass_flg = True 
  pass_flg = pass_flg and env_cfg is not None
  pass_flg = pass_flg and len(run_cfg_paths) > 0
  pass_flg = pass_flg and etype in project3_catalog[user]

  if etype is Type.grid and pass_flg:
    pass_flg = pass_flg and slice_path is not None
    
  if not pass_flg:
    logging.info("invalid input arguments for task")
  else:
    # main function
    for id_ex, run_cfg_path in enumerate(run_cfg_paths):
      name = os.path.basename(run_cfg_path)
      run_cfg = ConfigLoader().from_file(run_cfg_path)

      experment_f = project3_catalog[user][etype]

      logging.info(f'running experiment {id_ex + 1} with name {name} in mode {etype}')

      if etype is Type.run \
        or etype is Type.cv \
        or etype is Type.convert:
          experment_f(run_cfg, env_cfg)
      elif etype is Type.grid:
        slice_cfg = ConfigLoader().from_file(slice_path)
        experment_f(run_cfg, env_cfg, slice_cfg)

