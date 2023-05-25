#!/bin/bash -eux

docker run --name dylanlrrbio \
  -v ~/projects/dylanlrrb.github.io:/tf/notebooks \
  -v ~/projects/dylanlrrb.github.io/extensions:/root/.vscode-server/extensions \
  -v ~/.cache/torch/checkpoints:/root/.cache/torch/hub/checkpoints \
  -v ~/.cache/tensorflow_datasets:/root/tensorflow_datasets \
  -v ~/.cache/datasets:/root/datasets \
  -v ~/.cache/keras:/root/.keras \
  -v ~/.ngrok2:/root/.ngrok2 \
  -v ~/.aws:/root/.aws \
  --env-file ~/.docker-env \
  --gpus all -it -d \
  --rm gpu-image
