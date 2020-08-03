#!/usr/bin/env python
# Rui Zhang 7.2020
# rui.zhang@cern.ch

import json
import logging
import sys

def main():
    logging.basicConfig(level = logging.INFO)
    filenames = []
    try:
        filenames = sys.argv[1:]
    except:
        logging.fatal('python count_epoch.py Usage [json file name]')
    
    for filename in filenames:
        with open(filename) as f:
            data = json.load(f)
        
        try:
            logging.info(f'{filename} epochs {len(data["history"]["0:0:-"]["val_loss"])} val_loss {data["history"]["0:0:-"]["val_loss"][-10]}')
        except:
            logging.info(f'{filename} epochs {len(data["history"]["0:-:-"]["val_loss"])} val_loss {data["history"]["0:-:-"]["val_loss"][-10]}')

if __name__ == '__main__':
    main()
