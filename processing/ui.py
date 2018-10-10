"""User interface."""

from itertools import groupby
from os.path import dirname

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pynumparser
from dash.dependencies import Event, Input, Output, State
from dateutil.parser import parse as tparse

from app import app
from defaults import (AMP_RANGE_MAX, AMP_RANGE_MIN, DEF_BINS, DEF_RANGE_L,
                      DEF_RANGE_R)
from reducer import filter_points, get_points_cr, get_points_hist

FIG_1_TITLE = 'Points count rate (click to group by cr, select to show hist)'

app.css.append_css({
    "external_url":
    "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"
})

app.scripts.append_script({
    "external_url": "https://code.jquery.com/jquery-3.1.0.js"
})

app.scripts.append_script({
    "external_url":
    "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"
})

app.css.append_css(
    {"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.css.append_css(
    {"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Label('Voltage filter', htmlFor='voltage-filter'),
                dcc.Input(
                    type='text',
                    id='voltage-filter',
                    placeholder='Enter voltage values...',
                    value='16000',
                    className='form-control'
                ),
                html.Small(
                    'e.g. 16000-16500/50, 18000',
                    className='form-text text-muted')
            ], className='form-group'),
            html.Div([
                html.Label('Fill filter', htmlFor='fill-filter'),
                dcc.Input(
                    type='text',
                    id='fill-filter',
                    placeholder='Enter fill values...',
                    value='2017_05/Fill_1, 2017_05/Fill_2',
                    className='form-control'
                ),
                html.Small(
                    'e.g. 2017_05/Fill_1, 2017_05/Fill_2',
                    className='form-text text-muted')
            ], className='form-group'),
            html.Div([
                html.Label('Sets filter', htmlFor='sets-filter'),
                dcc.Input(
                    type='text',
                    id='sets-filter',
                    placeholder='Enter set values...',
                    value='1-100',
                    className='form-control'
                ),
                html.Small(
                    'e.g. 1-12, 15, 18',
                    className='form-text text-muted')
            ], className='form-group'),
            html.Div([
                html.Label('Window left border', htmlFor='l-border'),
                dcc.Input(
                    type='number',
                    id='l-border',
                    min=AMP_RANGE_MIN,
                    max=AMP_RANGE_MAX,
                    value=DEF_RANGE_L,
                    className='form-control'
                )
            ], className='form-group'),
            html.Div([
                html.Label('Window right border', htmlFor='r-border'),
                dcc.Input(
                    type='number',
                    id='r-border',
                    min=AMP_RANGE_MIN,
                    max=AMP_RANGE_MAX,
                    value=DEF_RANGE_R,
                    className='form-control'
                )
            ], className='form-group'),
            html.Button(
                'Apply', id='apply-button',
                className='btn btn-primary'),

        ], className='pull-right col-md-3',),
        html.Div([
            dcc.Graph(
                id='points-graph',
                figure={
                    'data': [],
                    'layout': {
                        'title': FIG_1_TITLE
                    }
                }
            ),
        ], className='col-md-9')
    ], className='row'),
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                    id='groups-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'title': '10% grouping'
                        }
                    }),
            ], className='col-md-6'),
            html.Div([
                dcc.Graph(
                    id='point-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'title': 'Point amplitudes histogram'
                        }
                    }),
            ], className='col-md-6')
        ], className='row'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Count rate percentage for grouping',
                               htmlFor='group-perc'),
                    dcc.Input(
                        type='number',
                        id='group-perc',
                        min=0,
                        max=100,
                        value=10,
                        className='form-control'
                    )
                ], className='form-group'),
            ], className='col-md-5 col-md-offset-1'),
            html.Div([
            ], className='col-md-6')
        ], className='row'),
    ], className='row'),
])


@app.callback(
    Output('point-graph', 'figure'),
    [Input('points-graph', 'selectedData')])
def _display_click_data(selectedData):

    if not selectedData:
        return {
            'data': [],
            'layout': {
                'title': 'Point amplitudes histogram'
            }
        }

    data = selectedData

    points = [d['customdata'] for d in data['points']]
    hists = get_points_hist(points)

    return {
        'data': [{
            'x': (bins[1:] + bins[:-1]) / 2,
            'y': hist,
            'name': point,
            'mode': 'lines',
            'showlegend': True,
            'line': {
                'shape': 'hvh'
            }
        } for point, (hist, bins) in zip(points, hists)],
        'layout': {
            'title': 'Point amplitudes histogram'
        }
    }


def _recalc(voltage, fill, sets, l_bord, r_bord):
    volt_range = pynumparser.NumberSequence().parse(voltage)
    fill_range = fill.split(', ')
    sets_range = pynumparser.NumberSequence().parse(sets)

    def filter_name(fp):
        return any((fill in fp for fill in fill_range)) and \
            any(('set_{}/'.format(s) in fp for s in sets_range))

    def filter_meta(meta):
        return any((
            int(v) == int(meta['external_meta']['HV1_value'])
            for v in volt_range))

    points = filter_points(filter_name, filter_meta)
    crs = get_points_cr((p[0] for p in points), l_bord, r_bord)

    combined = sorted([{
        'name': point[0],
        'meta': point[1],
        'cr': cr
    } for point, cr in zip(points, crs)],
        key=lambda o: tparse(o['meta']["params"]["start_time"]))

    sets = []
    for s, s_points in groupby(combined, lambda c: dirname(c['name'])):
        params = [{
            'x': tparse(o['meta']['params']['start_time']),
            'y': o['cr'],
            'text': 'set: {};<br>point index: {};<br>hv: {};'.format(
                s,
                o['meta']['external_meta']['point_index'],
                o['meta']['external_meta']['HV1_value']),
            'customdata': o['name'],
        } for o in s_points]

        sets.append({
            'name': s,
            'x': [p['x'] for p in params],
            'y': [p['y'] for p in params],
            'text': [p['text'] for p in params],
            'customdata': [p['customdata'] for p in params],
            'mode': 'markers',
            'type': 'scatter',
            'showlegend': True, **{}
        })

    return {
        'data': sets,
        'layout': {
            'title': FIG_1_TITLE
        }
    }


@app.callback(
    Output('points-graph', 'figure'),
    [],
    [State('voltage-filter', 'value'),
     State('fill-filter', 'value'),
     State('sets-filter', 'value'),
     State('l-border', 'value'),
     State('r-border', 'value'),
     State('points-graph', 'relayoutData'),
     State('points-graph', 'figure')],
    [Event('apply-button', 'click')])
def recalc_points(voltage, fill, sets, l_bord, r_bord, rdata, figure):
    """Apply filters and redraw points."""
    return _recalc(voltage, fill, sets, l_bord, r_bord)


def _filter_fig(fig, mean, perc=0.1):
    filtered = [
        (x, y, t) for y, (x, t) in (zip(fig['y'], zip(fig['x'], fig['text'])))
        if abs(y - mean) / mean < perc]
    out = {**fig}
    out['x'] = [f[0] for f in filtered]
    out['y'] = [f[1] for f in filtered]
    out['text'] = [f[2] for f in filtered]
    return out


@app.callback(
    Output('groups-graph', 'figure'),
    [Input('points-graph', 'clickData')],
    [State('group-perc', 'value'),
     State('points-graph', 'figure')])
def group(clickData, perc, figure):
    """Apply filters and redraw points."""
    if not clickData:
        return figure

    mean = np.mean([p['y'] for p in clickData['points']])

    figure['data'] = [
        d for d in (_filter_fig(f, mean, perc / 100 / 2)
                    for f in figure['data']) if d['x']]

    figure['layout']['title'] = '{}% grouping (mean = {})'.format(
        perc, np.round(mean, 4))

    return figure


server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)
