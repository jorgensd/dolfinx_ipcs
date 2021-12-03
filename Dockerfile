FROM ubuntu:20.04
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    git \
    wget \
    libmpich-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git clone https://github.com/jorgensd/dolfinx_ipcs/ && \
    pip3 install -r dolfinx_ipcs/requirements.txt