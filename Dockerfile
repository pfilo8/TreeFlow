# FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
FROM python:3.8-buster
USER root

RUN apt update
RUN apt install -y build-essential gcc g++ git openssh-server vim htop autoconf libboost-all-dev libtiff-dev curl \
    unzip libz-dev libpng-dev libjpeg-dev libopenexr-dev wget cmake sudo
RUN apt-get -y install tmux less

# Install SSH
RUN mkdir /var/run/sshd
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Change default port for SSH
RUN sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
RUN sed -i 's/Port 22/Port 4444/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 4444

COPY src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# Doesn't work in requirements
RUN pip install kedro-mlflow==0.7.6

# Add necessary paths to $PATH
RUN echo "export PYTHONPATH=/src:$PYTHONPATH" >> /root/.bashrc
RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> /root/.bashrc

# Create aliases
RUN echo 'alias jn="jupyter notebook --no-browser --ip=0.0.0.0 --allow-root"' >> ~/.bashrc
RUN echo 'alias tb="tensorboard --logdir=logs/ --host=0.0.0.0"' >> ~/.bashrc
RUN echo 'alias ll="ls -lah"' >> ~/.bashrc

# Set up language for Kedro
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

WORKDIR /src
USER root
CMD ["/usr/sbin/sshd", "-D"]
