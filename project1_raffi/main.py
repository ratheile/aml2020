from modules import ConfigLoader
import logging


def option1(arg):
  logging.info(f"option1 selected: {arg}")

def option2(arg):
  logging.info(f"option2 selected: {arg}")

options_dict = {
  'option1' : option1,
  'option2' : option2
}

def run(run_cfg, env_cfg):
  logging.warn(env_cfg)

  if run_cfg['preprocessing/outlier']:
    logging.warn("outlier detection ... done")

  if run_cfg['preprocessing/dim_red']:
    logging.warn("dimensionality reduction ... done")
  
  # both syntax work 
  # ['outer']['inner']
  # outer/inner
  # the first is primarely used if you iterate over things
  for i in run_cfg['array']:
    # seriously tho: never use this as a var name because of self ref!!!
    logging.error(f'this: {i["this"]} and that: {i["that"]}')

  # invoke different functions
  result_f = options_dict[run_cfg['task/variant']]
  logging.info(result_f(run_cfg['task']['cfg'])) 
  logging.info(result_f(run_cfg['task/cfg'])) 
  

  