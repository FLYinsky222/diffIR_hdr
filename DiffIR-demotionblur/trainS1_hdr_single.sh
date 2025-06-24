#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python DiffIR/train_hdr.py -opt options/train_DiffIRs1_hdr.yml --debug