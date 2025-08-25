#!/bin/env python3

import sys
from typing import List,Tuple
import numpy as np
from src.parse_xyz import read_xyz
from src._xyz import write_xyz
from src.DropFinder import DropletFinder


########################################################################
########################################################################
########################################################################

def droplet_xyz(
    outfn: str,
    box: np.ndarray, 
    periodic: List[bool], 
    atoms_traj: np.ndarray, 
    atoms_sigma: float, 
    atoms_cutoff: float, 
    meshspacing: float,
    nstd_cutoff: float = 3,
    cutoff_density: float = None,
    connect_range: float = None,
    extra_atoms: np.ndarray = None,
    drop_type='O',
    bulk_type='C',
    extra_types='N',
    ):
    
    atoms = atoms_traj[0]
    n = len(atoms)
    
    types = [bulk_type]*n + [drop_type]*n
    drop_traj = np.zeros((len(atoms_traj),2*n,3))
    drop_traj[:,:n] = atoms_traj
    drop_traj[:,n:] = atoms_traj
    hidepos = box[:,1]
    
    
    caln = 50
    calfreq = len(atoms_traj) // caln
    if caln == 0:
        calfreq = 1
    
    df = DropletFinder(
        atoms_traj[::calfreq],
        box,
        periodic,
        atoms_sigma,
        atoms_cutoff,
        meshspacing=meshspacing,
        nstd_cutoff=nstd_cutoff,
        cutoff_density=cutoff_density,
        connect_range=connect_range,
    )
    
    for s,atoms in enumerate(atoms_traj):
        if s%200 == 0:
            print(f'snapshot {s}')
        
        indroplet = df.droplet_atoms(atoms)
        for i in range(len(atoms)):
            if indroplet[i]:
                drop_traj[s,i] = hidepos
            else:
                drop_traj[s,n+i] = hidepos

    if extra_atoms is not None:
        f_traj = np.zeros((len(drop_traj),len(drop_traj[0])+len(extra_atoms[0]),3))
        f_traj[:,:len(extra_atoms[0])] = extra_atoms
        f_traj[:,len(extra_atoms[0]):] = drop_traj
        drop_traj = f_traj
        types = [extra_types]*len(extra_atoms[0]) + types
            
    data = {
        'pos' : drop_traj,
        'types' : types
    }
    write_xyz(outfn,data)



if __name__ == "__main__":

    periodic = [1,1,1]
    blocklen = 2
    atoms_sigma = 2
    atoms_cutoff = 5
    meshspacing = 1.0
    nstd_cutoff = 4
    connect_range = 1
    
    
    fn = sys.argv[1]
    fn = fn.replace('.xyz','')
    prot_type = sys.argv[2]
    
    cutoff_density = None
    if len(sys.argv) > 3:
        cutoff_density = float(sys.argv[3])
    
    data = read_xyz(fn)
    
    types = data['types']
    box = data['box']
    atoms_traj = data['pos']
    
    ##################################
    # limit trajectory
    print(atoms_traj.shape)
    atoms_traj = atoms_traj[::10]
    ##################################
    
    protids = [i for i, type in enumerate(types) if type == prot_type]
    otherids = [i for i, type in enumerate(types) if type != prot_type]
    
    prot_traj = atoms_traj[:,protids]
    other_traj = atoms_traj[:,otherids]

    outfn = fn+'_drop.xyz'
    droplet_xyz(
        outfn,box,periodic,
        prot_traj,atoms_sigma,
        atoms_cutoff,meshspacing,
        nstd_cutoff=nstd_cutoff,
        cutoff_density=cutoff_density,
        connect_range=connect_range,
        extra_atoms=other_traj,
        extra_types='H',)
    