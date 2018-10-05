# singnal-utils

- [ ] split requirements between core and scripts.


## Scripts usage samples

1. Convert Lan10-12PCI frames to events
    ```bash
    python3 convert_points.py /data/lan10 /data/lan10_processed/ **/*.df
    ```

2. Detect bad sets
    ```
    python scripts/analysis/detect_bad_sets.py ~/data/ 2017_05/Fill_2
    ```
