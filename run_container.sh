#!/bin/bash

docker run --name portfolio \
  -v ~/projects/dylanlrrb.github.io:/tf/notebooks \
  -v ~/projects/dylanlrrb.github.io/extensions:/root/.vscode-server/extensions \
  -v ~/.cache/torch/checkpoints:/root/.cache/torch/hub/checkpoints \
  --gpus all -it -d \
  --name portfolio \
  --rm gpu-image
