import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

import sys
import os
from pywt import wavedec, waverec

import heartpy as hp
import neurokit2 as nk

import matplotlib.pyplot as plt
import plotly.express as px

import seaborn as sns
import numpy as np
import pandas as pd
from copy import deepcopy
from modules import ConfigLoader

from biosppy.signals import ecg
import neurokit2 as nk


# %% some local testing:
env_cfg = ConfigLoader().from_file('env/env.yml')
# %%
print(env_cfg)

# %%
df_X = pd.read_csv(f"{env_cfg['datasets/project3/path']}/X_train.csv")
df_y = pd.read_csv(f"{env_cfg['datasets/project3/path']}/y_train.csv")
# df_X_u = pd.read_csv( f"{env_cfg['datasets/project3/path']}/X_test.csv")  # unlabeled

# %%
X = df_X.iloc[:, 1:]
y = df_y.iloc[:, 1:].values.ravel()
# X_u = df_X.iloc[:, 1:]

X_124 = X.iloc[y != 3]
X_3 = X.iloc[y == 3]

print("Data Loaded!!!")
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


available_classes = [0,1,2,3]

app.layout = \
html.Div([
dcc.Store(id='sync', data={
  'id' : 0,
  'features': ['filtered_nk2', 'filtered_bspy', 'quality','peaks_nk2', 'peaks_bspy']
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
    max=len(X),
    value=0,
    # marks={str(i): str(i) for i in range(len(X))},
    # step=None
  ),
  dcc.Input(id="samples-input", type="number", placeholder="", debounce=True),
  dcc.Checklist(
      id='features',
      options=[
          {'label': 'Show Filtered Neurokit2', 'value': 'filtered_nk2'},
          {'label': 'Show Filtered Biosppy', 'value': 'filtered_bspy'},
          {'label': 'Show Quality', 'value': 'quality'},
          {'label': 'Show Detected Peaks Neurokit2', 'value': 'peaks_nk2'},
          {'label': 'Show Detected Peaks Biosppy', 'value': 'peaks_bspy'},
      ],
      value=['filtered_nk2', 'filtered_bspy', 'quality','peaks_nk2', 'peaks_bspy']
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



@app.callback(
  Output('timeseries-plot', 'figure'), [
      Input('filter-class', 'value'),
      Input('sync', 'data')
  ])
def update_timeseries_plot(filter_class, data):
  if data is None:
      raise PreventUpdate

  crv = X.iloc[int(data['id'])] 
  crv = crv[~np.isnan(crv.values.astype(float))]
  sig_i_np = (crv.to_numpy()).ravel()

  # Convert data to plotting dataframe (plotly express)
  crv_dict = {
      'v': crv.values,
      't': time_scale(crv)
  }


  if 'filtered_bspy' in data['features']:
    out = ecg.ecg(signal=sig_i_np, sampling_rate=sample_rate, show=False)
    # (ts, filtered, rpeaks, templates_ts, 
    # templates, heart_rate_ts, heart_rate) = out
    _, filtered_bspy, peaks_bspy, _, _, _, _ = out

    crv_dict['filtered_bspy'] = filtered_bspy

  if 'filtered_nk2' in data['features']:
    out, info = nk.ecg_process(sig_i_np, sampling_rate=sample_rate)
    filtered_nk2 = out['ECG_Clean'].to_numpy().ravel()
    crv_dict['filtered_nk2'] = filtered_nk2
    # Signals is a dataframe


  
  if 'quality' in data['features']:
    print("computing quality ...")
    quality_nk2 = out['ECG_Quality']
    rate_nk2 = out['ECG_Rate']

  crv_df = pd.DataFrame(crv_dict)
  crv_y = list(crv_dict.keys())
  fig = px.line(crv_df, x='t', y=crv_y)


  if 'peaks_nk2' in data['features']:
    print("computing peaks ...")
    peaks_nk2 = out['ECG_R_Peaks']
    peaks_nk2 = peaks_nk2[peaks_nk2 == 1].index.tolist()
    peak_ts_nk2 = crv_dict['t'][peaks_nk2]
    peak_val_nk2 = crv[peaks_nk2]
    fig.add_scatter(x=peak_ts_nk2, y=peak_val_nk2, mode='markers')

  if 'peaks_bspy' in data['features']:
    print("computing peaks ...")
    peak_ts_bspy = crv_dict['t'][peaks_bspy]
    peak_val_bspy = crv[peaks_bspy]
    fig.add_scatter(x=peak_ts_bspy, y=peak_val_bspy, mode='markers')

  fig.update_layout(margin={
      'l': 40,
      'b': 40,
      't': 10,
      'r': 0
  }, hovermode='closest')

  return fig


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
      html.Td(f"{y[int(data['id'])]}")
    ])
  ])


# def create_time_series(dff, axis_type, title):

#   fig = px.scatter(dff, x='Year', y='Value')

#   fig.update_traces(mode='lines+markers')

#   fig.update_xaxes(showgrid=False)

#   fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

#   fig.add_annotation(x=0,
#                     y=0.85,
#                     xanchor='left',
#                     yanchor='bottom',
#                     xref='paper',
#                     yref='paper',
#                     showarrow=False,
#                     align='left',
#                     bgcolor='rgba(255, 255, 255, 0.5)',
#                     text=title)

#   fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

#   return fig


# @app.callback(dash.dependencies.Output('x-time-series', 'figure'), [
#     dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#     dash.dependencies.Input('crossfilter-xaxis-type', 'value')
# ])
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#   country_name = hoverData['points'][0]['customdata']
#   dff = df[df['Country Name'] == country_name]
#   dff = dff[dff['Indicator Name'] == xaxis_column_name]
#   title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#   return create_time_series(dff, axis_type, title)


# @app.callback(dash.dependencies.Output('y-time-series', 'figure'), [
#     dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#     dash.dependencies.Input('crossfilter-yaxis-type', 'value')
# ])
# def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#   dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#   dff = dff[dff['Indicator Name'] == yaxis_column_name]
#   return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
  app.run_server(debug=True)
