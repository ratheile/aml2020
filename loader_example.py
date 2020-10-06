from modules import ConfigLoader

import coloredlogs, logging
coloredlogs.install()

import argparse
import os

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)


if __name__ == "__main__":

  # Parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--env', type=file_path, default='env/env.yml',
    help='The environment yaml file.')

  args = parser.parse_args()
  env_cfg_path = args.env

  # Load configs
  env_cfg = ConfigLoader().from_file(env_cfg_path)
  
  logging.info(env_cfg['datasets/cifar10/path'])