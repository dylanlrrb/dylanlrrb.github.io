FROM tensorflow/tensorflow:latest-gpu-jupyter


WORKDIR /../root

RUN apt-get -y update
RUN apt-get -y install nodejs
RUN apt-get -y install npm
RUN npm install -g n
RUN n stable
RUN npm install -g ngrok

RUN curl -o .bash_profile https://raw.githubusercontent.com/dylanlrrb/dotfiles/linux/.bash_profile
RUN git clone https://github.com/dylanlrrb/dotfiles.git

RUN apt-get -y install tmux
RUN touch ~/.tmux.conf
RUN echo "set -g mouse on" >> ~/.tmux.conf

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
# need cython installed for cocoapi to install correctly
RUN pip3 install -U 'git+https://github.com/dylanlrrb/cocoapi.git#subdirectory=PythonAPI'

# chown -R root /tf/notebooks/

# ~/projects/dylanlrrb.github.io:/tf/notebooks
# ~/projects/dylanlrrb.github.io/extensions:/root/.vscode-server/extensions
# ~/.cache/torch/checkpoints:/root/.cache/torch/hub/checkpoints
# :/tmp/tfhub_modules