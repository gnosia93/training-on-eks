#!/bin/bash

# 1. 디렉토리 생성 (없을 경우)
mkdir -p data/wikihow

# 2. 데이터 다운로드
wget -P data/wikihow https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv
wget -P data/wikihow https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowSep.csv
