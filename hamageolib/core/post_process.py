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
from hamageolib.utils.vtk_utilities import get_pyvista_extension


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

    def process_piecewise(self, func_name, keys, **kwargs):
        '''
        Process the data piecewise. Put them into the dict of 
        self.combined to be merged later
        Inputs:
            func_name - name of the member function
            keys - key of the data structure
        '''
        # Set method to use 
        method = getattr(self, func_name)

        # Set options to method
        kwargs["save_file"] = False
        kwargs["update_attributes"] = False

        outputs = method(**kwargs)
        if isinstance(outputs, tuple):
            # multiple outputs
            assert(len(outputs) == len(keys))
            for i, key in enumerate(keys):
                assert(isinstance(outputs[i], pv.DataSet))
                if key not in self.combined:
                    self.combined[key] = []
                self.combined[key].append(outputs[i])
        elif isinstance(outputs, pv.DataSet):
            # only one output
            assert(len(keys)==1)
            if keys[0] not in self.combined:
                self.combined[keys[0]] = []
            self.combined[keys[0]].append(outputs)
        else:
            raise TypeError("Return value from function " + str(method) + " should be either tuple or pyvista.DataSet")

    def export_piecewise(self, i_piece, **kwargs):
        '''
        Export the results from a single piece
        Inputs:
            i_piece - number of piece
        '''
        from hamageolib.utils.vtk_utilities import pyvista_safe_save
        
        # Addtional options
        indent = 4

        # Save file
        start = time.time()

        for key, value in self.combined.items():
            assert(isinstance(value, list) and len(value)==1)
            extension = get_pyvista_extension(value[0])
            odir = os.path.join(self.pyvista_outdir, "p%02d" % i_piece)
            if not os.path.isdir(odir):
                os.mkdir(odir)
            ofile = os.path.join(odir, "%s_%05d.%s" % (key, self.pvtu_step, extension))
            pyvista_safe_save(value[0], ofile)
            print("%ssaved file for piece %d: %s" % (indent*" ", i_piece, ofile))
        
        end = time.time()
        print("%sPYVISTA_PROCESS: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def import_piecewise(self, i_piece, **kwargs):
        '''
        Import the results from a single piece
        Inputs:
            i_piece - number of piece
        '''
        indent = 4

        # import
        start = time.time()
        for key, value in self.__dict__.items():
            if value is None or isinstance(value, pv.DataSet):
                odir = os.path.join(self.pyvista_outdir, "p%02d" % i_piece)
                assert(os.path.isdir(odir))

                ofile_base = "%s_%05d" % (key, self.pvtu_step)
                ofile = find_file_by_stem(odir, ofile_base)
                if ofile is not None:
                    pv_obj = pv.read(ofile)

                    if key not in self.combined:
                        self.combined[key] = []
                    self.combined[key].append(pv_obj)
                    print("%sload attr " % (indent*" ") + key + " from piece " + str(i_piece))

        end = time.time()
        print("%sPYVISTA_PROCESS: %s takes %.1f s" % (indent * " ", func_name(), end - start))
    
    def combine_pieces(self):
        '''
        Merge the pv objects in self.combined
        '''
        indent = 4

        # merge objects in combined (dict)
        start = time.time()
        for key, value in self.combined.items():
            if len(value) == 0:
                merged = None
            else:
                merged = pv.merge(value, merge_points=True)
            setattr(self, key, merged)

        # reset self.combined
        self.combined = {}

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def reset_attrs(self, keys):
        '''
        reset attrs to None
        '''
        import gc
        for key in keys:
            setattr(self, key, None) 
    
    def write_key_to_file(self, key, filename_base, filetype):
        '''
        Write a single class attribute to file
        Inputs:
            key - name of the class attribute
            filename_base - basename of output file
            filetype - type of the file
        '''
        target = getattr(self, key)
        assert(target is not None)
        self.write_object_to_file(target, filename_base, filetype)

    def write_object_to_file(self, target, filename_base, filetype, **kwargs):
        '''
        Write a single class object to file
        Inputs:
            target - the target to write
            filename_base - basename of output file
            filetype - type of the file
        '''
        from hamageolib.utils.vtk_utilities import pyvista_safe_save

        indent = 4
        # check inputs
        assert(filetype in ["vtp", "vtu"])

        # export
        start = time.time()
        filename = "%s_%05d.%s" % (filename_base, self.pvtu_step, filetype)
        filepath = os.path.join(self.pyvista_outdir, filename)

        pyvista_safe_save(target, filepath)
        assert(os.path.isfile(filepath))
        print("%ssaved file %s" % (indent*" ", filepath))
        
        end = time.time()
        print("%sPYVISTA_PROCESS: %s takes %.1f s" % (indent * " ", func_name(), end - start))


class PYVISTA_PROCESS_WORKFLOW_ERROR(Exception):
    pass


# Utilities
def find_file_by_stem(directory, name_without_ext):
    directory = Path(directory)
    for file in directory.iterdir():
        if file.is_file() and file.stem == name_without_ext:
            return str(file.resolve())  # full absolute path
    return None