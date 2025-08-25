import numpy as np
import sys, os
from .parse_custom import read_custom_info
from .parse_datafile import datafile_read_box
from ._xyz import read_xyz, load_xyz

def read_xyz(fn,create_binary: bool = True, include_box: bool = True):
    
    basefn = fn.replace('.xyz','').replace('.data','').replace('.custom','')
    xyzfn = basefn + '.xyz'
    if not os.path.exists(xyzfn):
        raise FileNotFoundError(f"Error: File not found at {xyzfn}") 
    
    if create_binary:
        data = load_xyz(xyzfn,savenpy=True,loadnpy=True)
    else:
        data = read_xyz(xyzfn)

    if include_box:
        datafn = basefn + '.data'
        if not os.path.exists(datafn):
            datafn = datafn.replace('_equi','')
            if not os.path.exists(datafn):
                datafn = None
            
        customfn = basefn + '.custom'
        if not os.path.exists(customfn): 
            customfn = basefn + '.dat'
            if not os.path.exists(customfn):
                customfn = None
        
        if datafn is None and customfn is None:
            raise FileNotFoundError(f"Error: Neither custom file nore data file found. If box information is not required this error can be supressed by setting keyword argument include_box to False.") 

        if datafn is not None:
            box = datafile_read_box(datafn)
        else:
            info = read_custom_info(customfn)
            box = info['box']
        data['box'] = box
        
    return data
        
    

if __name__ == "__main__":
    
    fn = sys.argv[1]
    read_xyz(fn)