import os
import math
import time
import psutil
from pathlib import Path
import pyvista as pv
import numpy as np
from vtk import VTK_QUAD
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from hamageolib.utils.exception_handler import my_assert


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

    def read(self, pvtu_step, *,
             piece=None):
        
        start = time.time()

        self.pvtu_step = pvtu_step
        # check path of data
        if piece is None:
            filepath = os.path.join(self.data_dir, "solution-%05d.pvtu" % self.pvtu_step)
            my_assert(os.path.isfile(filepath), FileNotFoundError, "File %s is not found" % filepath)
        else:
            filepath = os.path.join(self.data_dir, "solution-%05d.%04d.vtu" % (self.pvtu_step, piece))
            my_assert(isinstance(piece, int) and piece >= 0, TypeError, "piece must be non-negative integar.")
            my_assert(os.path.isfile(filepath), FileNotFoundError, "File %s is not found" % filepath)
        
        end = time.time()
        print("PYVISTA_PROCESS:\n\tRead file %s" % (filepath))
        print("\ttakes %.1f s" % (end - start))

        # read data
        self.grid = pv.read(filepath)

class PYVISTA_PROCESS_WORKFLOW_ERROR(Exception):
    pass