import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

import argparse
import sys
import os
from pywt import wavedec, waverec

import heartpy as hp
import neurokit2 as nk

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import seaborn as sns
import numpy as np
import pandas as pd
from copy import deepcopy
from modules import ConfigLoader

from biosppy.signals import ecg
import neurokit2 as nk

import joblib
import logging 
import time


def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)


def create_app(D):
  I = { 
    "sample_id" : 0, # np.int64
    "class_id" : 1,  # np.int64
    "sig_i_np" : 2, # raw ecg  # ndarray
    "rpeaks_biosppy" : 3 , # ndarray
    "filtered_biosppy" : 4, # ndarray
    "signals": 5, # signals_neurokit # DataFrame
    "filter_mask": 6,
    'qc_success': 7,
    'is_flipped': 8
  }
  print("Data Loaded!!!")

  external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
  app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


  available_classes = [0,1,2,3]

  app.layout = \
  html.Div([
  dcc.Store(id='sync', data={
    'id' : 0,
    'features': ['original','filtered_nk2', 'filtered_bspy', 'quality','peaks_nk2', 'peaks_bspy', 'onsetoffset']
  }),
  html.Div([
    html.Div([
      dcc.Dropdown(id='filter-class',
        options=[{ 'label': i, 'value': i } for i in available_classes],
        value=0),
      ],
      style={
          'width': '49%',
          'display': 'inline-block'
      }
    ),

    html.Div([
      dcc.Dropdown(
        id='crossfilter-yaxis-column',
        options=[{ 'label': i, 'value': i } for i in available_classes],
        value=0),
      ],
      style={
        'width': '49%',
        'float': 'right',
        'display': 'inline-block' }
    )
    ],
    style={
      'borderBottom': 'thin lightgrey solid',
      'backgroundColor': 'rgb(250, 250, 250)',
      'padding': '10px 5px' }
  ),

  html.Div([
    dcc.Graph(
      id='timeseries-plot',
      hoverData={'points': [{
        'customdata': 'Japan'
      }]}
    )
    ],
    style={
      'width': '100%',
      'display': 'inline-block',
      'padding': '0 20' }
  ),

  html.Div([
    # dcc.Graph(id='x-time-series'),
    # dcc.Graph(id='y-time-series'),
    ],
    style={
      'display': 'inline-block',
      'width': '49%' }
  ),

  html.Div([
    dcc.Slider(
      id='samples-slider',
      min=0,
      max=len(D),
      value=0,
      # marks={str(i): str(i) for i in range(len(X))},
      # step=None
    ),
    dcc.Input(id="samples-input", type="number", placeholder="", debounce=True),
    dcc.Checklist(
        id='features',
        options=[
            {'label': 'Show Original', 'value': 'original'},
            {'label': 'Show Filtered Neurokit2', 'value': 'filtered_nk2'},
            {'label': 'Show Filtered Biosppy', 'value': 'filtered_bspy'},
            {'label': 'Show Quality', 'value': 'quality'},
            {'label': 'Show Detected Peaks Neurokit2', 'value': 'peaks_nk2'},
            {'label': 'Show Detected Peaks Biosppy', 'value': 'peaks_bspy'},
            {'label': 'Show All Onsets / Offsets', 'value': 'onsetoffset'},
        ],
        value=['original', 'filtered_nk2', 'filtered_bspy', 'quality','peaks_nk2', 'peaks_bspy']
    )  
    ],
    style={ 
      'width': '49%',
      'padding': '0px 20px 20px 20px' }
  ),

  html.Div(
    html.P(id='status')
  )

  ],
  )


  sample_rate = 300
  def time_scale(samples): return np.arange(samples.shape[0]) / sample_rate


  @app.callback([ Output("sync", "data")],
              [ Input("samples-input", "value"),
                Input("samples-slider", "value"),
                Input("features", "value")],
              [ State("sync", "data")]
              )
  def sync_input_value(input_value, slider_value, features, data):
    ctx = dash.callback_context

    if not ctx.triggered:
        input_id = 'Not triggered'
    else:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    print(f'sync callback {data}, trigger info {input_id}, new values: {[input_value, slider_value, features]}')

    if input_id == 'samples-input':
      data['id'] = input_value
    elif input_id == 'samples-slider' :
      data['id'] = slider_value
    elif input_id == 'features':
      data['features'] = features

    return [data]


  # @app.callback([ Output("samples-input", "value"),
  #                 Output("samples-slider", "value")], 
  #               [ Input("sync", "data")],
  #               [ State("samples-input", "value"), 
  #                 State("samples-slider", "value")])
  # def update_components(current_value, input_prev, slider_prev):
  #     # Update only inputs that are out of sync (this step "breaks" the circular dependency).
  #     input_value = current_value if current_value != input_prev else dash.no_update
  #     slider_value = current_value if current_value != slider_prev else dash.no_update
  #     return [input_value, slider_value]
  def compute_quality(a_really_long_var, a_really_other):
    pass


  @app.callback([
      Output('timeseries-plot', 'figure'), 
    ],
    [
      Input('filter-class', 'value'),
      Input('sync', 'data')
    ])

  def update_timeseries_plot(filter_class, data):
    if data is None:
        raise PreventUpdate
    
    D_id = D[int(data['id'])]
    sig_i_np = D_id[I['sig_i_np']]
    crv = sig_i_np
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25])
    time_ax = time_scale(crv)


    if 'original' in data['features']:
      fig.add_trace(
        go.Scatter(x=time_ax, y=crv, name='original'),
        row=1, col=1
      )

    if 'filtered_bspy' in data['features']:
      # out = ecg.ecg(signal=sig_i_np, sampling_rate=sample_rate, show=False)
      # (ts, filtered, rpeaks, templates_ts, 
      # templates, heart_rate_ts, heart_rate) = out
      # _, filtered_bspy, peaks_bspy, _, _, _, _ = out
      filtered_bspy = D_id[I['filtered_biosppy']]
      peaks_bspy = D_id[I['rpeaks_biosppy']]

      fig.add_trace(
        go.Scatter(x=time_ax, y=filtered_bspy, name='filtered_bspy'),
        row=1, col=1
      )

    if 'filtered_nk2' in data['features']:
      out = D_id[I['signals']]
      # out, info = nk.ecg_process(sig_i_np, sampling_rate=sample_rate)
      filtered_nk2 = out['ECG_Clean'].to_numpy().ravel()
    
      fig.add_trace(
        go.Scatter(x=time_ax, y=filtered_nk2, name='filtered_nk2'),
        row=1, col=1
      )

    if 'quality' in data['features']:
      print("computing quality ...")
      quality_nk2 = out['ECG_Quality']
      filter_mask = D_id[I['filter_mask']]
      rate_nk2 = out['ECG_Rate']
      
      
      fig.add_trace(
        go.Scatter(x=time_ax, y=rate_nk2, name='rate_nk2'),
        row=2, col=1
      )
      fig.add_trace(
        go.Scatter(x=time_ax, y=filter_mask.astype(float), name='quality_nk2'),
        row=3, col=1
      )

      fig.add_trace(
        go.Scatter(x=time_ax, y=quality_nk2, name='quality_nk2'),
        row=3, col=1
      )

    def plot_points(
      metric_name, y=crv, color='black', size=12,
      mask=None, color_false='red',
      row=1):
      metric = out[metric_name]
      marker_index = metric[metric == 1].index.tolist()
      marker_ts = time_ax[marker_index]

      args = dict(
        y=y[marker_index],
        name=metric_name, mode='markers',
        marker=dict(
          color=color,
          size=size,
          line=dict(
            color='MediumPurple',
            width=3
          )
        )
      )

      if mask is not None:
        marker_mask = mask[marker_index]
        colors = [color if i == True else color_false for i in marker_mask]
        args['colors'] = colors

      fig.add_trace(
        go.Scatter(x=marker_ts, **args),
        row=row, col=1
      )


    def plot_lines(metric_start, metric_end, color='black'):
      a = out[metric_start]
      b = out[metric_end]
      a = a[a == 1].index.tolist()
      b = b[b == 1].index.tolist()
      a = time_ax[a]
      b = time_ax[b]

      for i in a:   
        fig.add_vline(i, row=1, col=1)

    if 'onsetoffset' in data['features']:
      # Index(['ECG_Raw', 'ECG_Clean', 'ECG_Rate', 'ECG_Quality', 'ECG_R_Peaks',
      #        'ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks',
      #        'ECG_P_Onsets', 'ECG_T_Offsets', 'ECG_Phase_Atrial',
      #        'ECG_Phase_Completion_Atrial', 'ECG_Phase_Ventricular',
      #        'ECG_Phase_Completion_Ventricular'],
      #       dtype='object')
      plot_points('ECG_Q_Peaks', color='green')


      plot_points('ECG_Q_Peaks',y=quality_nk2, color='green', row=3)
      plot_points('ECG_Q_Peaks',y=rate_nk2, color='green', row=2)

      plot_points('ECG_S_Peaks', color='blue')

      plot_points('ECG_P_Peaks', color='orange')
      plot_points('ECG_P_Onsets', color='orange')

      plot_points('ECG_T_Peaks', color='red')
      plot_points('ECG_T_Offsets', color='red')
      # plot_lines('ECG_P_Onsets', 'ECG_T_Offsets')




    if 'peaks_bspy' in data['features']:
      print("computing peaks ...")
      peak_ts_bspy = time_ax[peaks_bspy]
      peak_val_bspy = crv[peaks_bspy]
      fig.add_trace(
        go.Scatter(
          x=peak_ts_bspy, y=peak_val_bspy, name='peaks_bspy', mode='markers',
          marker=dict(
              symbol='x',
              color='purple',
              size=15,
              line=dict(
                  color='violet',
                  width=3
              )
          )),
        row=1, col=1
      )

    if 'peaks_nk2' in data['features']:
      print("computing peaks ...")
      plot_points('ECG_R_Peaks', color='black')
      


    fig.update_layout(
      margin={ 'l': 40, 'b': 40, 't': 10, 'r': 0 },
      hovermode='closest',
      height=700,
      yaxis2=dict(
        rangemode='tozero'
      )
    )

    return [fig]


  @app.callback(
      Output('status', 'children'), [
          Input('sync', 'data')
      ])
  def update_info(data):
    if data is None:
        raise PreventUpdate
    return html.Table([
      html.Tr([html.Th("Info"), html.Th("Value") ]),
      html.Tr([
        html.Td("showing graph for timeseries id:"),
        html.Td(f"{data['id']}"),
      ]),
      html.Tr([
        html.Td("Class"),
        html.Td(f"{D[int(data['id']), I['class_id']]}")
      ]),
      html.Tr([
        html.Td("Quality Check"),
        html.Td(f"{D[int(data['id']), I['qc_success']]}")
      ]),
      html.Tr([
        html.Td("Flipping"),
        html.Td(f"{D[int(data['id']), I['is_flipped']]}")
      ])
    ])

  return app




if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  input_grp = parser.add_mutually_exclusive_group(required=True)

  parser.add_argument('--env', type=file_path, default='env/env.yml',
  help='The environment yaml file.')

  input_grp.add_argument('--joblib', type=file_path,
  help='The joblib file.')

  args = parser.parse_args()

  # %% some local testing:
  env_cfg = ConfigLoader().from_file(args.env)
  print(env_cfg)

  D = joblib.load(args.joblib)

  app = create_app(D)
  app.run_server(debug=False)
