"""UWSGI application with plugins."""
from os import environ

import dash
from dotenv import load_dotenv
from flask_caching import Cache

load_dotenv()

app = dash.Dash(__name__)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://{}/cache'.format(
        environ['REDIS_HOSTNAME'])
})
