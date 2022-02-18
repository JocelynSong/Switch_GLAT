#!/bin/bash

export http_proxy=http://sys-proxy-rd-relay.byted.org:3128
export https_proxy=http://sys-proxy-rd-relay.byted.org:3128


python3 setup.py build_ext --inplace
pip install .
pip install sacremoses
