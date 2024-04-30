FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /../root

# needed if public key changes for outdated tf versions
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install nodejs
RUN apt-get -y install npm
RUN npm install -g n
RUN n stable
RUN npm install -g ngrok
RUN apt-get -y install tmux
RUN touch ~/.tmux.conf
RUN echo "set -g mouse on" >> ~/.tmux.conf

RUN curl -o .bash_profile https://raw.githubusercontent.com/dylanlrrb/dotfiles/linux/.bash_profile
RUN git clone https://github.com/dylanlrrb/dotfiles.git
RUN source .bash_profile

COPY ./requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

WORKDIR /../tf/notebooks
RUN git config --global --add safe.directory /tf/notebooks

# need cython installed for cocoapi to install correctly
# RUN pip3 install -U 'git+https://github.com/dylanlrrb/cocoapi.git#subdirectory=PythonAPI'

# chown -R root /tf/notebooks/
# docker system prune -a