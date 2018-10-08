"""Detector model config."""
from os import environ
from os.path import abspath, dirname, join

from dotenv import load_dotenv
from kombu import Exchange, Queue
from kombu.serialization import register

from serializer import pack, unpack

load_dotenv()

register(
    'msgpack-numpy', pack, unpack,
    content_type='application/x-msgpack-numpy',
    content_encoding='binary',
)

broker_url = 'amqp://{}:{}@{}//'.format(
    environ['RABBITMQ_DEFAULT_USER'],
    environ['RABBITMQ_DEFAULT_PASS'],
    environ['RABBITMQ_HOSTNAME']
)
result_backend = 'redis://{}/0'.format(
    environ['REDIS_HOSTNAME'])

task_serializer = 'msgpack-numpy'
result_serializer = 'msgpack-numpy'

accept_content = ['msgpack-numpy']
