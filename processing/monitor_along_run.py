# -*- coding: utf-8 -*-
import json
from datetime import datetime
from itertools import chain
from os.path import join, relpath
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html
import dfparser
import numpy as np
from dash.dependencies import Event, Input, Output
from dateutil.parser import parse as tparse
from tqdm import tqdm

from utils import _get_points_in_set, _get_sets_in_fill, _lan_amps, _lan_amps_f

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

DATA_PATH = '/media/chernov/data/data/lan10_processed/'
FILL_PATH = '2017_05/Fill_2/'

DEF_VOLTAGE = 16050
DEF_RANGE_L = 0
DEF_RANGE_R = 4096
DEF_BINS = 64

SETS = _get_sets_in_fill(join(DATA_PATH, FILL_PATH))


def get_cr(point, range_l=DEF_RANGE_L, range_r=DEF_RANGE_R):
    """Calculate point count rate."""
    _, meta, data = dfparser.parse_from_file(point)

    amps = _lan_amps(data)
    event_len = meta['params']['b_size'] / meta['params']['sample_freq']
    time_total = event_len * meta['params']['events_num']
    counts = amps[np.logical_and(amps < range_r, amps > range_l)].shape[0]

    cr = counts / time_total

    return {
        'time': tparse(meta['params']['start_time']),
        'cr': cr,
        'filename': point,
        'tooltip': 'index: {}'.format(meta['external_meta']['point_index'])
    }


def get_hist(point, range_l=DEF_RANGE_L, range_r=DEF_RANGE_R, bins=DEF_BINS):
    """Calculate point histogram."""
    amps = _lan_amps_f(point)
    hist, bins = np.histogram(amps, range=(range_l, range_r), bins=bins)
    return (bins[1:] + bins[:-1]) / 2,  hist


def get_sets_data(
        voltage=DEF_VOLTAGE,
        range_l=DEF_RANGE_L, range_r=DEF_RANGE_R, bins=DEF_BINS):
    for set_ in tqdm(SETS, desc='loading sets'):
        points = _get_points_in_set(set_)
        crs = [
            get_cr(p) for p in tqdm(points, leave=False) if str(voltage) in p]
        yield {
            'x': [p['time'] for p in crs],
            'y': [p['cr'] for p in crs],
            'text': [p['tooltip'] for p in crs],
            'name': relpath(set_, join(DATA_PATH, FILL_PATH)),
            'customdata': [p['filename'] for p in crs],
            'type': 'scatter',
            'mode': 'markers',
            'showlegend': True,
            'marker': {
                'size': 15
            }
        }


def get_voltages():
    points = list(chain(*(_get_points_in_set(s) for s in SETS)))

    voltages = []
    for point in tqdm(points, 'indexing voltages'):
        _, meta, _ = dfparser.parse_from_file(point, nodata=True)
        voltages.append(meta['external_meta']['HV1_value'])

    return sorted(list(set(voltages)))


VOLTAGES = get_voltages()

app.layout = html.Div(children=[
    html.H1(children='Hello Dash!!!'),

    html.Div([
        dcc.Dropdown(
            id='voltage-dropdown',
            options=[{'label': v, 'value': v} for v in VOLTAGES],
            value=DEF_VOLTAGE
        ),
        dcc.Graph(
            id='fill-graph',
            figure={
                'data': list(get_sets_data(DEF_VOLTAGE)),
                'layout': {
                    'title': 'Monitor points count rate'
                }
            }
        ),
    ],
        style={'width': '59%', 'display': 'inline-block'}
    ),

    html.Div([
        dcc.Graph(
            id='point-graph',
            figure={
                'data': [],
                'layout': {
                    'title': 'Point amplitudes histogram'
                }
            }),
        dcc.RangeSlider(
            id='my-range-slider',
            min=0,
            max=4096,
            marks={
                i: {'label': i} for i in range(0, 4096, 512)
            },
            value=[0, 4096]
        ),
    ],
        style={'width': '39%', 'display': 'inline-block'},
    ),
    html.Div([
        dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
        html.Pre(id='click-data'),
    ], className='three columns'),
])


@app.callback(
    Output('point-graph', 'figure'),
    [Input('fill-graph', 'selectedData'), Input('fill-graph', 'hoverData')])
def _display_click_data(selectedData, hoverData):
    points = []
    if selectedData:
        points = selectedData['points']
    elif hoverData:
        points = hoverData['points']

    shapes = []
    for point in points:
        bins, hist = get_hist(point['customdata'])
        shapes.append({
            'x': bins,
            'y': hist,
            'name': relpath(point['customdata'], join(DATA_PATH, FILL_PATH)),
            'mode': 'lines',
            'showlegend': True,
            'line': {
                'shape': 'hvh'
            }
        })

    return {
        'data': shapes,
        'layout': {
            'title': 'Point amplitudes histogram'
        }
    }


@app.callback(Output('fill-graph', 'figure'),
              [Input('voltage-dropdown', 'value')])
def update_metrics(value):
    style = {'padding': '5px', 'fontSize': '16px'}
    return {
        'data': list(get_sets_data(value)),
        'layout': {
            'title': 'Monitor points count rate'
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
