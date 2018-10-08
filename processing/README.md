Fill quality control processing


celery -A worker worker --config celeryconfig --concurrency=8


echo never > /sys/kernel/mm/transparent_hugepage/enabled
sysctl vm.overcommit_memory=1
