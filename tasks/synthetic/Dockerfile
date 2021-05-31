FROM ubuntu
MAINTAINER Hiroshi Kajino <kajino@jp.ibm.com>

# initial setup
RUN useradd -m -d /home/docker -s /bin/bash docker -u 1001
RUN echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
#RUN chown docker:docker /home/docker

RUN apt-get -qq update
RUN apt-get install -y tzdata
RUN apt-get install -y gcc g++ git
RUN apt-get install -y aria2 curl wget bzip2
RUN apt-get install -y make unzip zlib1g-dev
RUN apt-get install -y sqlite openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev
RUN apt-get install -y tk-dev
RUN apt-get install -y emacs
RUN apt-get install -y liblzma-dev
RUN apt-get install -y liblapack-dev libblas-dev gfortran
RUN apt-get install -y libffi-dev
RUN apt-get install -y texlive-full
RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc
RUN apt-get install -y language-pack-ja
RUN update-locale LANG=ja_JP.UTF-8

# change user
USER docker
ENV HOME /home/docker
WORKDIR /home/docker
VOLUME /home/docker/tasks

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.profile
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
RUN pyenv install 3.9.0
RUN pyenv global 3.9.0
RUN pyenv init - >> ~/.profile
RUN pyenv init - >> ~/.bash_profile

# setup default python libraries
RUN pip install pip --upgrade

ENV LANG=ja_JP.UTF-8
ENV OMP_NUM_THREADS 1

WORKDIR /home/docker


RUN git clone https://github.com/kanojikajino/luigine.git
WORKDIR /home/docker/luigine
RUN pip install .
WORKDIR /home/docker/

RUN git clone https://github.com/ibm-research-tokyo/diffsnn.git
WORKDIR /home/docker/diffsnn
RUN pip install .
WORKDIR /home/docker/
WORKDIR /home/docker/diffsnn/tasks/synthetic/

ENV LANG ja_JP.UTF-8
