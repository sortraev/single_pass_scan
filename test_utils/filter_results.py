#!/usr/bin/env python
import numpy as np
import sys

if len(sys.argv) != 3:
  print("Usage: "+sys.argv[0]+" <filename_in> <filename_out>")
  quit()

filename_in  = sys.argv[1]
filename_out = sys.argv[2]

results = np.loadtxt(filename_in, delimiter=',')           # load benchmark results.

results = results[np.lexsort(np.rot90(results))]           # sort results lexicographically by N and bandwidth measurement.

results = np.flipud(results)                               # flip results before extracting max bandwidth for each N.

_, indices = np.unique(results[:,0], return_index = True)  # for each N, extract the row with the maximum bandwidth.
results = results[indices]

np.savetxt(filename_out, results, fmt='%1.3f')
