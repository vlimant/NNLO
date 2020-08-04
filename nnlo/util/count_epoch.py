#!/usr/bin/env python
# Rui Zhang 7.2020
# rui.zhang@cern.ch

import json
import logging
import sys
import pandas as pd

def main():
    logging.basicConfig(level = logging.INFO)
    filename, rows_list = [], []
    try:
        filenames = sys.argv[1:]
    except:
        logging.fatal('python count_epoch.py Usage [json file name]')
    
    for filename in filenames:
        with open(filename) as f:
            data = json.load(f)
        name = filename.split('_')[1]
        dic = {
            'file': filename,
            'ranks': int(name[name.find('n')+1:name.find('g')]),
            'trainTime': data["train_time"],
        }
        try:
            dic['val_loss'] = data["history"][r"0:0:-"]["val_loss"][-10]
            dic['val_accuracy'] = data["history"][r"0:0:-"]["val_accuracy"][-10]
            dic['epochs'] = len(data["history"][r"0:0:-"]["val_loss"])
        except:
            dic['val_loss'] = data["history"][r"0:-:-"]["val_loss"][-10]
            dic['val_accuracy'] = data["history"][r"0:-:-"]["val_accuracy"][-10]
            dic['epochs'] = len(data["history"][r"0:-:-"]["val_loss"])
        rows_list.append(dic)

    df = pd.DataFrame(rows_list).sort_values('ranks')
    logging.info(f'\n{df}')

if __name__ == '__main__':
    main()
