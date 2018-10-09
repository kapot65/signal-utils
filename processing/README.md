Fill quality control processing

## Installation
1. Build docker services `docker-compose up`



celery -A worker worker --config celeryconfig --concurrency=$WORKER_CONCURRENCY


echo never > /sys/kernel/mm/transparent_hugepage/enabled
sysctl vm.overcommit_memory=1
