#!/bin/env python3

import sys,glob,os
from typing import List,Tuple
import numpy as np
from src.parse_xyz import read_xyz
from src.DropFinder import DropletFinder
        

if __name__ == "__main__":


    periodic = [1,1,0]
    blocklen = 2
    atoms_sigma = 2
    atoms_cutoff = 5
    meshspacing = 1.0
    nstd_cutoff = 4
    
    
    fn = sys.argv[1]
    prot_type = sys.argv[2]

    fn = fn.replace('.xyz','')
    data = read_xyz(fn)
            
    types = data['types']
    box = data['box']
    atoms_traj = data['pos']
    
    df = DropletFinder(
        atoms_traj[::20],
        box,
        periodic,
        atoms_sigma,
        atoms_cutoff,
        meshspacing=meshspacing,
    )
    
    for i in range(len(atoms_traj)):
        dropnums = df.largest_droplet([atoms_traj[i]])
        print(f'{i*10000}: {dropnums[0]}')