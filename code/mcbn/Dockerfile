FROM ubuntu:16.04

WORKDIR /mcbn
ADD . /mcbn

RUN apt-get update
RUN apt -y install python-pip
RUN apt-get -y install python-tk
RUN pip install virtualenv
