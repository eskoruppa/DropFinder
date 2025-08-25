#!/bin/env python3

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

"""
########################################################
    
    
    specs = load_xyz(filename)
    specs = read_xyz(filename)
    
        readxyz always reads the xyz file while loadxyz accesses the 
        binary if it exists and creates it if it doesn't such that access
        will be accelerated next time.
            
########################################################
"""

XYZ_NPY_EXT = "_xyz.npy"


def load_xyz(
    filename: str, savenpy: bool = True, loadnpy: bool = True
) -> Dict[str, Any]:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")
    fnpy = os.path.splitext(filename)[0] + XYZ_NPY_EXT
    if (
        loadnpy
        and os.path.isfile(fnpy)
        and os.path.getmtime(fnpy) >= os.path.getmtime(filename)
    ):
        xyz = dict()
        print(f"loading positions from '{fnpy}'")
        xyz["pos"] = np.load(fnpy)
        xyz["types"] = read_xyz_atomtypes(filename)
        return xyz
    xyz = read_xyz(filename)
    if savenpy:
        _save_xyz_binary(fnpy, xyz["pos"])
    return xyz


def load_pos_of_type(
    filename: str, selected_types: List[str], savenpy: bool = True, loadnpy: bool = True
) -> np.ndarray:
    xyz = load_xyz(filename, savenpy=savenpy, loadnpy=savenpy)
    ids = [id for id in range(len(xyz["types"])) if xyz["types"][id] in selected_types]
    data = np.array([snap[ids] for snap in xyz["pos"]])
    return data


def _linelist(line: str) -> List[str]:
    return [elem for elem in line.strip().split(" ") if elem != ""]

def read_xyz(filename: str) -> Dict[str, Any]:
    print(f"reading '{filename}'")
    dims = find_xyz_dimensions(filename)
    print(f"{dims[0]} snapshots with {dims[1]} monomers.")
    data = np.empty((dims) + (3,))
    with open(filename) as f:
        line = f.readline().rstrip()
        # define rec
        rec = str(line)
        ####
        snap = -1
        while line != "":
            ll = _linelist(line)
            if line != rec and len(ll) >= 4: # and 'ATOMS' not in line: # and ll[0] != "Atoms.":
                snap += 1
                if snap >= len(data):
                    break
                bp = 0
                while len(ll) >= 4:
                    data[snap, bp] = [float(ft) for ft in ll[1:4]]
                    bp += 1
                    line = f.readline()
                    ll = _linelist(line)
            line = f.readline()
    xyz = dict()
    xyz["pos"] = data
    xyz["types"] = read_xyz_atomtypes(filename)
    return xyz

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

def find_xyz_dimensions(filename: str) -> Tuple:
    with open(filename) as f:
        line = f.readline().rstrip()
        # define rec
        rec = str(line)
        lrec = len(rec)
        ####
        ll = _linelist(line)
        while len(ll) < 4 or line[:lrec] == rec:
            line = f.readline().replace("\n", "")
            ll = _linelist(line)
        nbp = 0
        while len(ll) >= 4 and line[:lrec] != rec:
            nbp += 1
            line = f.readline()
            ll = _linelist(line)
    with open(filename) as f:
        num_snap = 0
        for line in f:
            if rec == line[:lrec]:
                num_snap += 1
    return (num_snap, nbp)


def read_xyz_atomtypes(filename: str) -> List:
    data = list()
    with open(filename) as f:
        line = f.readline()
        num = 0
        types = list()
        while line != "":
            ll = _linelist(line)
            if len(ll) >= 4 and ll[0] != "Atoms.":
                num += 1
                if num > 1:
                    break
                while len(ll) >= 4:
                    types.append(ll[0])
                    line = f.readline()
                    ll = _linelist(line)
            line = f.readline()
    return types


def write_xyz(
    outfn: str, data: dict, add_extension: bool = True, append: bool = False
) -> None:
    """
    Writes configuration to xyz file

    Parameters
    ----------
    outfn : string
        name of xyz file

    """
    if ".xyz" not in outfn.lower() and add_extension:
        outfn += ".xyz"

    pos = data["pos"]
    types = data["types"]

    if "timesteps" in data.keys() and len(data["timesteps"]) == len(data["pos"]):
        timesteps_provided = True
    else:
        timesteps_provided = False

    nbp = len(pos[0])
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(outfn, mode) as f:
        for s, snap in enumerate(pos):
            f.write("%d\n" % nbp)
            if timesteps_provided:
                f.write("Atoms. Timestep: %d\n" % (data["timesteps"][s]))
            else:
                f.write("Atoms. Timestep: %d\n" % (s))
            for i in range(nbp):
                f.write(
                    "%s %.4f %.4f %.4f\n"
                    % (types[i], snap[i, 0], snap[i, 1], snap[i, 2])
                )


def _save_xyz_binary(outname: str, data: np.ndarray) -> None:
    if os.path.splitext(outname)[-1] == ".npy":
        outn = outname
    else:
        outn = outname + ".npy"
    np.save(outn, data)
    
def isxyz(fn: str) -> bool:
    if os.path.splitext(fn)[-1].lower() == '.xyz':
        return True
    return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python %s fin fout" % sys.argv[0])
        sys.exit(0)
    fin = sys.argv[1]
    fout = sys.argv[2]
    xyz = load_xyz(fin)
    types = read_xyz_atomtypes(fin)
    print(f"number of atoms = {len(types)}")
