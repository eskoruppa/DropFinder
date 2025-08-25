#!/bin/env python3

import sys
from typing import List,Tuple
import numpy as np
# from numba import jit, njit
from .conditional_numba import conditional_numba
from .parse_xyz import read_xyz
from ._xyz import write_xyz

_DROPFINDER_PLT_IMPORTED = True
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print('matplotlib not installed. Deactivate plotting functions')
    _DROPFINDER_PLT_IMPORTED = False

# np.set_printoptions(linewidth = 250,precision=3,edgeitems=10)

########################################################################
########################################################################
########################################################################

class DropletFinder:
    
    def __init__(
        self,
        atoms_traj: np.ndarray, 
        box: np.ndarray, 
        periodic: List[bool], 
        atoms_sigma: float, 
        atoms_cutoff: float, 
        meshspacing: float = None,
        nstd_cutoff: float = 4,
        gaussian_width_nsig: float = 1.5,
        cutoff_density: float = None,
        connect_range: float = None,
        atoms_per_bin: int = 50,
        num_calibrations: int = 50,
        print_status: bool = False,
        ):
        
        self.print_status = print_status
        
        self.atoms_traj = atoms_traj
        if not isinstance(self.atoms_traj,np.ndarray):
            self.atoms_traj = np.array(self.atoms_traj)
        if len(self.atoms_traj.shape) == 3:
            self.atoms_tray = np.array([self.atoms_traj])
            
        self.box = np.array(box,dtype=np.double)
        self.periodic = np.array(periodic,dtype=np.int32)
        self.atoms_sigma = atoms_sigma
        self.atoms_cutoff = atoms_cutoff
        self.meshspacing = meshspacing
        self.nstd_cutoff = nstd_cutoff
        self.gaussian_width_nsig = gaussian_width_nsig
        
        self.cutoff_density = cutoff_density
        self.atoms_per_bin = atoms_per_bin
        self.connect_meshrange = self.set_connect_mesh_range(connect_range)
        self.num_calibrations = num_calibrations
        
        self.atom_density_hists = []
        
        if self.print_status: print('calibrating cutoff density')
        calib_every = len(atoms_traj) // self.num_calibrations
        if calib_every == 0 and len(atoms_traj) > 0:
            calib_every = 1
        for s in range(0,len(atoms_traj),calib_every):
            atoms = atoms_traj[s]
            mesh = density_mesh(self.box,self.periodic,atoms,self.atoms_sigma,self.atoms_cutoff,self.meshspacing,gaussian_width_nsig=self.gaussian_width_nsig)
            hist = self.atom_density_hist(atoms,mesh)
            self.atom_density_hists.append(hist)
            if self.print_status: print(f'calib {s}: {self.find_cutoff_density(hist)}')
        
        self.cutoff_density = cutoff_density
        if self.cutoff_density is None:
            self.cutoff_density = np.mean([self.find_cutoff_density(hist) for hist in self.atom_density_hists])

        if self.print_status: print('calculating droplet properties')
        self.droplet_registers = []
        for s,atoms in enumerate(atoms_traj):
            mesh = density_mesh(self.box,self.periodic,atoms,self.atoms_sigma,self.atoms_cutoff,self.meshspacing,gaussian_width_nsig=self.gaussian_width_nsig)
            dropmesh = self.connect_droplets(mesh)
            droplet_register = self.droplet_atoms(atoms,dropmesh)
            self.droplet_registers.append(droplet_register)
            if self.print_status: print(f' num droplets = {len(np.unique(self.droplet_atoms(atoms,dropmesh,25).flatten()))-1}')  
        
    def droplet_atoms(
        self,
        atoms: np.ndarray,
        dropmesh: np.ndarray = None,
        min_atoms_per_dropet: int = None,
    ):
        if dropmesh is None:
            mesh = density_mesh(self.box,self.periodic,atoms,self.atoms_sigma,self.atoms_cutoff,self.meshspacing,gaussian_width_nsig=self.gaussian_width_nsig)
            dropmesh = self.connect_droplets(mesh)
    
        dL  = self.box[:,1] - self.box[:,0]
        dlo = self.box[:,0]
        Nd = np.array(dropmesh.shape)
        indroplet = np.zeros(len(atoms),dtype=np.int32)
        for i,atom in enumerate(atoms):
            atom = (atom - dlo) %dL + dlo
            coords = ((atom - dlo) * Nd // dL).astype(np.int32) 
            indroplet[i] = dropmesh[coords[0],coords[1],coords[2]]
        
        if min_atoms_per_dropet is not None:
            indroplet = self.filter_droplets(indroplet, min_atoms_per_dropet)
        return indroplet  
    

    def filter_droplets(self,indroplet: np.ndarray, min_atoms_per_dropet: int):
        indroplet = np.copy(indroplet)
        for dropid in range(1,np.max(indroplet)+1):
            if (indroplet == dropid).sum() < min_atoms_per_dropet:
                indroplet[indroplet==dropid] = 0
        return indroplet    


    def largest_droplet(self, atoms_traj: np.ndarray = None):
        droplet_nums = self.atoms_per_droplet(atoms_traj=atoms_traj)
        largest_drops = []
        for dn in droplet_nums:
            if len(dn) == 0:
                largest = 0 
            else:
                largest = np.max(dn)
            largest_drops.append(largest)
        return np.array(largest_drops)

    def num_droplets(
        self,
        atoms_traj: np.ndarray = None,
        min_atoms_per_dropet: int = 0):
        droplet_atoms = self.atoms_per_droplet(atoms_traj=atoms_traj)

        drop_num = []
        for j in range(len(droplet_atoms)):
            drop_num.append(len([nat for nat in droplet_atoms[j] if nat > min_atoms_per_dropet]))
        return drop_num 
        

    def atoms_per_droplet(self, atoms_traj: np.ndarray = None):
        if atoms_traj is None:
            droplet_registers = self.droplet_registers
        else:
            droplet_registers = []
            for s,atoms in enumerate(atoms_traj):
                mesh = density_mesh(self.box,self.periodic,atoms,self.atoms_sigma,self.atoms_cutoff,self.meshspacing,gaussian_width_nsig=self.gaussian_width_nsig)
                dropmesh = self.connect_droplets(mesh)
                indroplet = self.droplet_atoms(atoms,dropmesh)
                droplet_registers.append(indroplet)
        
        droplet_nums = []
        for droplet_register in droplet_registers:
            nums = [(droplet_register == did).sum() for did in range(1,np.max(droplet_register)+1)]
            droplet_nums.append(nums)
        # droplet_nums = [[(droplet_register == did).sum() for did in range(1,np.max(droplet_register)+1)] for droplet_register in droplet_registers]
        return droplet_nums
    

    def has_droplet(self, atoms_traj: np.ndarray = None, min_atoms_per_dropet: int = 25):
        largest_drops = self.largest_droplet(atoms_traj=atoms_traj)
        hasdrop = np.array(largest_drops >= min_atoms_per_dropet)
        return hasdrop


    
    def droplet_mesh(
        self,
        atoms: np.ndarray, 
        connect_meshrange: int = None,
        cutoff_density: float = None,
        ):
        
        if len(atoms.shape) > 2:
            raise ValueError('droplet_mesh expects a single snapshot. Multiple snapshots detected')
        mesh = density_mesh(self.box,self.periodic,atoms,self.atoms_sigma,self.atoms_cutoff,self.meshspacing,gaussian_width_nsig=self.gaussian_width_nsig)
        return self.connect_droplets(mesh,connect_meshrange=connect_meshrange,cutoff_density=cutoff_density)
    
    def connect_droplets(
        self,
        mesh: np.ndarray, 
        connect_meshrange: int = None,
        cutoff_density: float = None,
        ):
        
        if connect_meshrange is None:
            connect_meshrange = self.connect_meshrange
        if cutoff_density is None:
            cutoff_density = self.cutoff_density

        dropmesh = mesh > cutoff_density
        dropmesh = -dropmesh.astype(np.int32)
        
        dropid = 0
        dropstart = first_minus_one_coord(dropmesh)
        while dropstart is not None:
            dropid += 1
            dropmesh = check_neighbors(dropmesh,dropstart,dropid,self.periodic,rge=connect_meshrange)
            dropstart = first_minus_one_coord(dropmesh)
        return dropmesh
     
     
    
    def set_connect_mesh_range(self,connect_range: float):   
        dL  = self.box[:,1] - self.box[:,0]
        Nd = (np.ceil(dL / self.meshspacing)).astype(np.int32)
        ddx = dL / Nd
        if connect_range is None:
            self.connect_meshrange = 1
        else:
            self.connect_meshrange = int(np.min(connect_range / ddx))
        return self.connect_meshrange

    def atom_density_hist(
        self,
        atoms: np.ndarray,
        mesh: np.ndarray
        ):
        atomdens = cal_atomdens(self.box,atoms,mesh)
        hist,edges = np.histogram(atomdens,bins=len(atoms)//self.atoms_per_bin,density=True)
        hist_dens = 0.5*(edges[1:]+edges[:-1])
        fhist = np.array([hist_dens,hist]).T
        return fhist


    def find_cutoff_density(
        self,
        fhist: np.ndarray
        ):
        hist_dens = fhist[:,0]
        hist      = fhist[:,1]
    
        cdf = np.cumsum(hist) 
        first_peak_id = np.argmax(hist[:len(hist)//2])
        # first point less than 5%
        frac = 0.02
        low_id = -1
        for i in range(first_peak_id+1,len(hist)):
            if hist[i]/cdf[i] <= frac or hist[i]/hist[i-1] < 0.2:
                low_id = i
                break
    
        if low_id == -1:
            return hist_dens[-1]
    
        low_id = low_id + np.argmin(hist[low_id:int(np.min([low_id*2+2,len(hist)-1]))])
        phist = np.copy(hist[:low_id+1])
        phist /= phist.sum()
        phist_dens = hist_dens[:low_id+1]
        mean = (phist * phist_dens).sum()
        var  = (phist * (phist_dens - mean)**2).sum()
        std = np.sqrt(var)
        
        cutoff_dens = np.max([np.min([hist_dens[first_peak_id]+self.nstd_cutoff*std,hist_dens[-1]]),hist_dens[low_id]])
        cutoff_dens_id = find_nearest_index(hist_dens,cutoff_dens)
        
        # cutoff_dens_id = np.max([hist_dens[first_peak_id]+4*std,low_id])
        # cutoff_dens = hist[cutoff_dens_id]
        
        # # Create the figure and a 3D Axes
        # fig = plt.figure(figsize=(8.6/2.54, 10/2.54))
        # ax1  = fig.add_subplot(211)
        # ax2  = fig.add_subplot(212)
        
        # # ax1.hist(atomdens,bins=len(self.atoms)//50,density=True)
        # # ax.hist(atomdens,density=True)
        # ax1.plot(hist_dens,hist,color='black',lw=1)
    
        # ax1.scatter(hist_dens[first_peak_id],hist[first_peak_id],s=20,edgecolor='red',color='None')
        # ax1.scatter(hist_dens[low_id],hist[low_id],s=20,edgecolor='blue',color='None')
        # ax1.scatter(hist_dens[first_peak_id]+self.nstd_cutoff*std,10,s=20,edgecolor='green',color='None')
        
        # ncdf = cdf / np.sum(hist)
        # ax2.plot(hist_dens,ncdf,color='black',lw=1)
    
        # fig.savefig(density_profile_fn+'.png',dpi=300,transparent=False)
        # plt.close()  
        return cutoff_dens
    
    def plot_atom_densities(
        self,
        basefn: str, 
        png: bool = True,
        pdf: bool = False,
        svg: bool = False):
        if not _DROPFINDER_PLT_IMPORTED:
            print('Plotting functionality deactivated because matplotlib is not installed')
            return
        
        for i,fhist in enumerate(self.atom_density_hists):
            
            outfn = basefn + f'_#{i}'
            
            hist_dens   = fhist[:,0]
            hist        = fhist[:,1] 
            cdf = np.cumsum(hist) 
            ncdf = cdf / np.sum(hist)
        
            # Create the figure and a 3D Axes
            fig = plt.figure(figsize=(8.6/2.54, 10/2.54))
            ax1  = fig.add_subplot(211)
            ax2  = fig.add_subplot(212)
            axes = [ax1,ax2]
            
            ax1.plot(hist_dens,hist,color='black',lw=1)
            # ax1.fill_between(hist_dens,0,hist,color='blue',alpha=0.3)
            
            nearest_id = find_nearest_index(hist_dens,self.cutoff_density)
            hist_bulk = fhist[:nearest_id+1]
            hist_drop = fhist[nearest_id:]
            
            ax1.fill_between(hist_bulk[:,0],0,hist_bulk[:,1],color='blue',alpha=0.5)
            ax1.fill_between(hist_drop[:,0],0,hist_drop[:,1],color='red',alpha=0.5)
            
            ylim = [0,np.max(hist)*1.05]
            maxplotdens = np.max([np.max(hist_dens),self.cutoff_density])
            xlim = [np.min(hist_dens)-maxplotdens*0.02,maxplotdens*1.02]
            
            ax1.plot([self.cutoff_density,self.cutoff_density],ylim,lw=1.4,ls='--',color='black')
            ax2.plot(hist_dens,ncdf,color='black',lw=1)
            
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
            ax1.set_ylim(ylim)
            
            axlinewidth  = 0.5
            axtick_major_width  = 0.5
            axtick_major_length = 1.6
            axtick_minor_width  = 0.5
            axtick_minor_length = 1
            tick_pad        = 2
            tick_labelsize  = 5
            label_fontsize  = 6
            label_fontweight= 'bold'
            legend_fontsize = 6
            for ax in axes:
                ###############################
                # set major and minor ticks
                ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad)
                ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length)
                    # print(zl)
                    # print(forces)
                    
                ###############################
                ax.xaxis.set_ticks_position('both')
                # set ticks right and top
                ax.yaxis.set_ticks_position('both')
                
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(axlinewidth)


            ax1.set_xlabel('Local Atom Density',fontsize=label_fontsize,fontweight=label_fontweight,labelpad=2)
            ax2.set_xlabel('Local Atom Density',fontsize=label_fontsize,fontweight=label_fontweight,labelpad=2)
            ax1.set_ylabel('Probability Density',fontsize=label_fontsize,fontweight=label_fontweight,labelpad=2)
            ax2.set_ylabel('Cummulative Distribution',fontsize=label_fontsize,fontweight=label_fontweight,labelpad=2)

            ax1.xaxis.set_label_coords(0.5,-0.08)
            ax2.xaxis.set_label_coords(0.5,-0.08)
            ax1.yaxis.set_label_coords(-0.07,0.5)
            ax2.yaxis.set_label_coords(-0.07,0.5)

            plt.subplots_adjust(left=0.1,
                                right=0.98,
                                bottom=0.06,
                                top=0.98,
                                wspace=0.3,
                                hspace=0.2)
            
            if png: fig.savefig(outfn+'.png',dpi=300,transparent=False)
            if pdf: fig.savefig(outfn+'.pdf',dpi=300,transparent=True)
            if svg: fig.savefig(outfn+'.svg',dpi=300,transparent=True)
        
            plt.close()        
        





########################################################################
########################################################################
########################################################################
# extra methods

# @njit(cache=True, parallel=False)
@conditional_numba
def density_mesh(
    box: np.ndarray, 
    periodic: np.ndarray,
    atoms: np.ndarray, 
    atom_sigma: float, 
    atom_cutoff: float, 
    meshspacing: float,
    gaussian_width_nsig: float = 1.5,
):

    dims = len(box)
    dL  = box[:,1] - box[:,0]
    dlo = box[:,0]
    Nd = (np.ceil(dL / meshspacing)).astype(np.int32)
    ddx = dL / Nd
    
    # edgezone 
    edge_n = np.ceil(atom_cutoff / ddx).astype(np.int32)
    Ndtot = Nd + 2*edge_n

    # full mesh
    # mesh = np.zeros(tuple(Ndtot),dtype=np.float64)
    mesh = np.zeros((Ndtot[0],Ndtot[1],Ndtot[2]),dtype=np.float64)
    
    # build distance mask
    mask_n = edge_n * 2 + 1
    mids   = np.copy(edge_n)
    # d2mask = np.zeros(mask_n)
    d2mask = np.zeros((mask_n[0],mask_n[1],mask_n[2]))
    for x in range(mask_n[0]):
        for y in range(mask_n[1]):
            for z in range(mask_n[2]):
                v = (np.array([x,y,z]) - mids)*ddx
                v2 = np.dot(v,v)
                d2mask[x,y,z] = v2 
    
    # build density mask
    sig = atom_sigma*gaussian_width_nsig
    sig2 = sig**2
    dv = np.prod(ddx)
    pre = 1./(2*np.pi*sig2)**(dims/2)
    maskrho = pre * np.exp(-d2mask / (2*sig2)) * dv
    # normalize (dv technically not necessary)
    maskrho = maskrho / np.sum(maskrho)

    # compute mesh
    for atom in atoms:
        # coords = (atom - dlo) * Nd // dL + edge_n
        
        # lammps coords may be slightly out of bounds
        atom = (atom - dlo) %dL + dlo
         
        lower = ((atom - dlo) * Nd // dL).astype(np.int32) 
        upper = lower + mask_n

        # mesh[tuple(slice(lo, hi) for lo, hi in zip(lower, upper))] += maskrho        
        mesh[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]] += maskrho

    # fold periodic copies
    if periodic[0]:
        mesh[-2*edge_n[0]:-edge_n[0]] += mesh[:edge_n[0]]
        mesh[edge_n[0]:2*edge_n[0]]   += mesh[-edge_n[0]:]
    mesh = mesh[edge_n[0]:-edge_n[0]]
    
    if periodic[1]:
        mesh[:,-2*edge_n[1]:-edge_n[1]] += mesh[:,:edge_n[1]]
        mesh[:,edge_n[1]:2*edge_n[1]]   += mesh[:,-edge_n[1]:]
    mesh = mesh[:,edge_n[1]:-edge_n[1]]
    
    if periodic[2]:
        mesh[:,:,-2*edge_n[2]:-edge_n[2]] += mesh[:,:,:edge_n[2]]
        mesh[:,:,edge_n[2]:2*edge_n[2]]   += mesh[:,:,-edge_n[2]:]
    mesh = mesh[:,:,edge_n[2]:-edge_n[2]]
        
    return mesh

@conditional_numba
def cal_atomdens(box,atoms,mesh):
    dL  = box[:,1] - box[:,0]
    dlo = box[:,0]
    Nd = np.array(mesh.shape)
    atomdens = np.zeros(len(atoms))
    for i,atom in enumerate(atoms):
        atom = (atom - dlo) %dL + dlo
        coords = (atom - dlo) * Nd // dL
        atomdens[i] = mesh[int(coords[0]),int(coords[1]),int(coords[2])]
    return atomdens

# @njit(cache=True, parallel=False)
@conditional_numba
def check_neighbors(
    dropmesh: np.ndarray, 
    start: np.ndarray, 
    did: int, 
    periodic: np.ndarray, 
    rge:int=1):
    
    xl, yl, zl = dropmesh.shape
    max_size = xl * yl * zl
    stack = np.empty((max_size, 3), np.int64)
    sp = 0

    x0, y0, z0 = start[0], start[1], start[2]
    dropmesh[x0, y0, z0] = did
    # stack[0, 0], stack[0, 1], stack[0, 2] = x0, y0, z0
    stack[0] = start
    sp = 1

    iter = 0
    while sp > 0:
        sp -= 1
        x0, y0, z0 = stack[sp]
        # x0, y0, z0 = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        
        iter+=1
        if iter%1000 ==0:
            unassigned = len(np.argwhere(dropmesh==-1))
            if unassigned == 0:
                break
            
        # Examine neighbors
        for dx in range(-rge, rge+1):
            x = x0 + dx
            if x < 0 or x >= xl:
                if periodic[0]:
                    x %= xl
                else:
                    continue
            for dy in range(-rge, rge+1):
                y = y0 + dy
                if y < 0 or y >= yl:
                    if periodic[1]:
                        y %= yl
                    else:
                        continue
                for dz in range(-rge, rge+1):
                    z = z0 + dz
                    if z < 0 or z >= zl:
                        if periodic[2]:
                            z %= zl
                        else:
                            continue

                    if dropmesh[x, y, z] == -1:
                        dropmesh[x, y, z] = did     
                        stack[sp, 0] = x
                        stack[sp, 1] = y
                        stack[sp, 2] = z
                        sp += 1
    return dropmesh


def first_minus_one_coord(A: np.ndarray):
    mask = (A.ravel() == -1)
    if not mask.any():
        return None
    flat_idx = mask.argmax()
    return np.unravel_index(flat_idx, A.shape)


def find_nearest_index_simple(a: np.ndarray, x: float) -> int:
    return np.abs(a - x).argmin()


def find_nearest_index(a: np.ndarray, x: float) -> int:
    """
    Given a sorted 1D numpy array `a` and a value `x`,
    return the index of the element in `a` closest to `x`.
    """
    # find insertion point
    i = a.searchsorted(x)
    
    # clamp to ends
    if i == 0:
        return 0
    if i == len(a):
        return len(a) - 1
    
    # pick the closer of a[i-1] and a[i]
    prev_diff = x - a[i - 1]
    next_diff = a[i] - x
    if prev_diff <= next_diff:
        return i - 1
    else:
        return i
    
    
# DEPRICATED
def density_distribution(atoms,box,gridsize):
    dL  = box[:,1] - box[:,0]
    dlo = box[:,0]
    Nd = (np.ceil(dL / gridsize)).astype(np.int32)

    volmesh = np.zeros((Nd[0],Nd[1],Nd[2]),dtype=np.float64)
    for i,atom in enumerate(atoms):
        atom = (atom - dlo) %dL + dlo
        co = ((atom - dlo) * Nd // dL).astype(np.int32) 
        volmesh[co[0],co[1],co[2]] += 1
    return volmesh

########################################################################
########################################################################
########################################################################

        

if __name__ == "__main__":

    box = np.array([[0,60],[0,80],[0,800]])
    periodic = [1,1,1]
    blocklen = 2
    atoms_sigma = 2
    atoms_cutoff = 5
    gridsize = 1.0
    nstd_cutoff = 4
    
    
    fn = sys.argv[1]
    fn = fn.replace('.xyz','')
    
    cutoff_density = None
    if len(sys.argv) > 2:
        cutoff_density = float(sys.argv[2])
    
    data = read_xyz(fn)

    box = data['box']
    atoms_traj = data['pos']
    atoms_traj = atoms_traj[:,400:]
    dna_traj = atoms_traj[:,:400]
    meshspacing = gridsize
    connect_range = 1
    
    freq = 100
    
    df = DropletFinder(
        atoms_traj[::freq],
        box,
        periodic,
        atoms_sigma,
        atoms_cutoff,
        meshspacing=meshspacing,
    )
    
    df.largest_droplet()
    hasdrop = df.has_droplet()
    droplets_contained = len(hasdrop[hasdrop==True]) > len(hasdrop[hasdrop==False])
    print(f'droplets found = {droplets_contained}')
        
    
    # freq = 200
    # outfn = f'figs/{fn.split("/")[-1].replace(".xyz","")}_snap{i}'
    # print(f'step {i}')
    # atoms = pos[i]
    # dropmesh = droplet_finder(
    #     box,
    #     periodic,
    #     atoms,
    #     atoms_sigma,
    #     atoms_cutoff,
    #     gridsize,
    #     gaussian_width_nsig=1.5
    # )
    # sys.exit()


    
    base_outfn = f'figs/{fn.split("/")[-1].replace(".xyz","")}'
    
    df.plot_atom_densities(base_outfn)

