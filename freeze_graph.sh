#!/bin/bash
frozen_graph="$1.frz"
python -m tensorflow.python.tools.freeze_graph --input_graph $1 --input_checkpoint $2 --output_graph $frozen_graph --output_node_names=$3 --input_binary=true
