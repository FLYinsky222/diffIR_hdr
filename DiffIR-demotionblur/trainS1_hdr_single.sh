#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python DiffIR/train.py -opt options/train_DiffIRS1_hdr.yml --launcher none