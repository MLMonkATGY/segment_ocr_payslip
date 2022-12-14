#!/bin/bash
conda env export --no-builds >environment.yml &&
    pip list --format=freeze >requirements.txt
