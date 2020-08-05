#!/usr/bin/env python
# Rui Zhang 7.2020
# rui.zhang@cern.ch

import sys

def main():
    command = sys.argv[1]
    argv = sys.argv[2:]
    if command.lower() == 'mnist':
        from nnlo.data.get_mnist import main
        main(argv)
    elif command.lower() == 'cifar10':
        from nnlo.data.get_cifar10 import main
        main(argv)
    else:
        raise RuntimeError('Unknown command: {}'.format(command))

if __name__ == '__main__':
    main()
