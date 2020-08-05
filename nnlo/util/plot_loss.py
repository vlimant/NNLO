#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

import json
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plotLoss(data, outname='', variable='loss'):
    nGPU = len(data["history"].keys())
    for irank, values in data["history"].items():
        irank = irank.split(':')[0]
        if irank.startswith('0'):
            plt.plot(values['val_'+variable], linestyle='-', label=f'R{irank}: val')
            if variable in values:
                plt.plot(values[variable], linestyle=':', label=f'R{irank}: train')
        else:
            plt.plot(values[variable], linestyle='--', label='')

    plt.title(outname)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{variable}_{outname}.pdf', format='pdf')
    logging.info(f'Save {variable}_{outname}.pdf')
    plt.clf()

def main():
    logging.basicConfig(level = logging.INFO)
    filename, rows_list = [], []
    try:
        filenames = sys.argv[1:]
    except:
        logging.fatal('Usage: python plot_loss.py [json file name]')

    for filename in filenames:
        with open(filename) as f:
            logging.info(f'Read {filename}')
            data = json.load(f)
            outname = filename.replace('.json', '')
            for variable in ['loss', 'accuracy']:
                plotLoss(data, outname, variable)


if __name__ == '__main__':
    main()
