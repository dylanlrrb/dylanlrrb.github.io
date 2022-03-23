FROM tensorflow/tensorflow:latest-gpu-jupyter


WORKDIR /../root

RUN apt-get -y update
RUN apt-get -y install nodejs
RUN apt-get -y install npm
RUN npm install -g ngrok
RUN npm install -g n
RUN n stable

RUN curl -o .bash_profile https://raw.githubusercontent.com/dylanlrrb/dotfiles/linux/.bash_profile
RUN git clone https://github.com/dylanlrrb/dotfiles.git

COPY ./requirements.txt .
RUN pip install -r requirements.txt

# chown -R root /tf/notebooks/

# ~/projects/dylanlrrb.github.io:/tf/notebooks
# ~/projects/dylanlrrb.github.io/extensions:/root/.vscode-server/extensions
# ~/.cache/torch/checkpoints:/root/.cache/torch/hub/checkpoints
# :/tmp/tfhub_modules