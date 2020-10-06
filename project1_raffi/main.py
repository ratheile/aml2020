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

def run(rcfg):
  # logging.info(rcfg)

  if rcfg['preprocessing/outlier']:
    logging.warn("outlier detection ... done")

  if rcfg['preprocessing/dim_red']:
    logging.warn("dimensionality reduction ... done")
  
  # both syntax work 
  # ['outer']['inner']
  # outer/inner
  # the first is primarely used if you iterate over things
  for i in rcfg['array']:
    # seriously tho: never use this as a var name because of self ref!!!
    logging.error(f'this: {i["this"]} and that: {i["that"]}')

  # invoke different functions
  result_f = options_dict[rcfg['task/variant']]
  logging.info(result_f(rcfg['task']['cfg'])) 
  

  