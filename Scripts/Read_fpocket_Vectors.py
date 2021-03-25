"""
Use the .ff file and extract the FFvectors from the Query environments -> DEPRECATED - /Collector/Data/ contains dataframe
For EnsembleModels
"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import numpy as np


from argparse import ArgumentParser

parser = ArgumentParser(description="Read FF")
parser.add_argument("feature_file", type=str, help="input - .ff - change")
args = parser.parse_args()

def ReadFF(f):
        # Read in the raw ff file
    ffraw=[]
    with open(f, 'r') as fffile:
        ffdata = fffile.readlines()
        for li in ffdata:
            if li.startswith("Env_"):
                ffraw.append(str(li.split("\n")[0]))

        # Process the str
    fflol = []
    for item in ffraw:
        try:
            renamed = [item.split("\t")[0].split("_")[1],                                            # Conformer index
                 "%s.%s" %(item.split("\t")[-1].split(":")[0], item.split("\t")[-1].split(":")[1]),   # Center Type 
                                         item.split("\t")[-1].split(":")[3],                  # Chain
                                         item.split("\t")[-1].split(":")[2],                  # Residue
                                         item.split("\t")[-1],                                # Full Annotation
                                         item.split("\t")[-5],                                # X coord
                                         item.split("\t")[-4],                                # Y coord
                                         item.split("\t")[-3]]                                # Z coord
            renamed.extend(item.split("\t")[1:-6])                                                   # 480 FEATURE properties
        except IndexError: 
       # TODO Strangely the readline becomes unstable for some ~1 to 2 strings but not all the strings in CDK5, which the problem does not exist for CDk2. I need someone to reproduce this error... The item is printed below 
            print(f, item)

        fflol.append(tuple(renamed))
    return fflol

temp = ReadFF(args.feature_file)
for j, i in enumerate(temp):
    temp[j] = list(i)

matr = np.empty((len(temp),480))
for j, _ in enumerate(matr):
    matr[j] = list(map(float, temp[j][8:]))

pickle.dump(matr, open(args.feature_file[:-2]+'_property.pvar', "wb"))
