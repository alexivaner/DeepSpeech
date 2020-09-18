#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if __name__ == '__main__':
    try:
        from deepspeech_training import train as ds_train
    except ImportError:
        print('Training package is not installed. See training documentation.')
        raise

    ds_train.run_script()
