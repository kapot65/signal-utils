### Usage
1. Mount Lan10 and  MADC data folders

      sudo mkdir /mnt/lan10-server-data/
      sudo sshfs -o allow_other trdat@192.168.111.1:/data /mnt/lan10-server-data/

      sudo mkdir /mnt/madc-server-data/
      sudo sshfs -o allow_other trdat@192.168.111.1:/home/trdat/data /mnt/madc-server-data/

2. Mount Lan10 computer data

      sudo mkdir /mnt/lan10-comp-data/
      sudo sshfs -o allow_other chernov@192.168.111.23:/data /mnt/lan10-comp-data
