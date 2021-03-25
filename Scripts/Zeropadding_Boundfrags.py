"""
Use the boundfrags.txt file and applies zero-padding to create a square matrix -> for the KB
For EnsembleModels
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser(description="Zero Pad File")
parser.add_argument("bound_frags", type=str, help="input - boundfrags.txt - change")
args = parser.parse_args()

with open(args.bound_frags, "r") as f:
    linecount = 0
    maxcount = 0
    for line in f:
        if len(line.split()) > maxcount:
            maxcount = len(line.split())
        linecount+=1

matr = np.zeros((linecount, maxcount), dtype=int)
with open(args.bound_frags, "r") as f:
    for j, line in enumerate(f):
        temp = line.split()
        matr[j, 0:len(temp)] = list(map(int, temp))
np.savetxt(args.bound_frags[:-4] + '_zeros.txt', matr, delimiter=',', fmt ='%i')
print(matr.shape)