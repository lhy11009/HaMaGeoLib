import os
import math
import time
import psutil
from pathlib import Path
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator
from vtk import VTK_QUAD


# todo_pyv
class PYVISTA_PROCESS():

    """
    Class for post-processing spherical shell data from geodynamic simulations using PyVista.
    
    Author: Haoyuan Li
    License: MIT
    """

    def __init__(self, data_dir, *,
                 pyvista_outdir=None):
        
        # data_directory
        self.data_dir = data_dir

        if pyvista_outdir is None:
            self.pyvista_outdir = os.path.join(self.data_dir, "..", "pyvista_outputs")
        else:
            self.pyvista_outdir = pyvista_outdir
        
        if not os.path.isdir(self.pyvista_outdir):
            os.mkdir(self.pyvista_outdir)

        # Initialize global variables 
        self.pvtu_step = None
        self.grid = None