FROM ubuntu:16.04
MAINTAINER Arwa Sheraky <arwasheraky.github.io>

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install numpy pandas argparse matplotlib seaborn plotly sklearn
