#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python DiffIR/train_hdr_recover.py -opt options/train_DiffIRs1_recover_hdr.yml --debug