#!/usr/bin/env python

import numpy as np

guesses = np.random.randint(0,3,size=10);
L = ['barren land', 'trees', 'grassland', 'none']
s = ','.join([L[t] for t in guesses])
f = open('landuse.csv', 'w')
f.write(s)
f.close()
