FROM ubuntu:16.04
MAINTAINER Disa Mhembere

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -o Dpkg::Options::="--force-confold" --force-yes -y upgrade

RUN apt-get -y install \
        build-essential \
        libboost-all-dev \
        libssl-dev libxml2-dev\
        libcurl4-openssl-dev\
        libnuma-dbg libnuma-dev libnuma1\
        libgoogle-perftools-dev \
        r-base-core

WORKDIR /home/ubuntu/
RUN Rscript -e "install.packages('devtools', dependencies=TRUE, repos='http://cran.rstudio.com/')"

#RUN Rscript -e "install.packages('devtools', dependencies=TRUE, repos='http://cran.rstudio.com/'); require(devtools); install_github('flashxio/knorR')"
# Enter bash shell
ENTRYPOINT ["bash"]
