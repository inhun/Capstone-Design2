import time
import sys
import os
import math

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def crop_box(frame, stats, i):
    x, y, w, h, s = stats[i]

    if s < 1000:
        return None

    if w/h > 5:
        return None
    if h/w > 5:
        return None
        
    if x+w >= 1920:
        if y + h >= 1080:
            dst = frame[y:1080, x:1920].copy()
        else:
            dst = frame[y:y+h, x:1920].copy()
    else:
        if y + h >= 1080:
            dst = frame[y:1080, x:x+w].copy()
        else:
            dst = frame[y:y+h, x:x+w].copy()
    return dst