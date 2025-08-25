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
    
    min_snap = 100
    considered_snap = 50
    
    path = sys.argv[1]
    prot_type = sys.argv[2]
    
    fns = glob.glob(path+'/*.xyz')
    for fn in fns:        
        fn = fn.replace('.xyz','')
        try:
            data = read_xyz(fn)
        except FileNotFoundError as e:
            print(fn)
            print(e)
            continue
            
        types = data['types']
        box = data['box']
        atoms_traj = data['pos']
        
        protids = [i for i, type in enumerate(types) if type == prot_type]
        otherids = [i for i, type in enumerate(types) if type != prot_type]
        
        atoms_traj = atoms_traj[:,protids]
        other_traj = atoms_traj[:,otherids]

        print(f'{len(atoms_traj)} snapshots found')
        if len(atoms_traj) >= min_snap:

            startid = len(atoms_traj) // 2
            check_every = len(atoms_traj[startid:]) // considered_snap
            if check_every == 0 and len(atoms_traj[startid:]) > 0:
                check_every = 1
            
            try:
                df = DropletFinder(
                    atoms_traj[startid::check_every],
                    box,
                    periodic,
                    atoms_sigma,
                    atoms_cutoff,
                    meshspacing=meshspacing,
                )
                
                num_drop = df.num_droplets(min_atoms_per_dropet=25)
                mean_num = np.mean(num_drop)
                print(f'Mean number of droplets: {mean_num}')
            except ValueError:
                mean_num = -1
        else:
            mean_num = -1
            
        outfn = fn + '.dropnum' 
        with open(outfn,'w') as f:
            f.write(f'{mean_num}')     
    