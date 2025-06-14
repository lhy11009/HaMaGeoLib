# import xml colormaps to paraview from ScientificColourMaps
# Usage:
#   1. substitute the SCM_dir with the directory to SCM map
#   2. run with:
#       paraview --script load_xml.py
#### import the simple module from the paraview
from paraview.simple import *
import os
import sys
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def import_xmls(_dir):
    '''
    Import xmls files
    '''
    for subdir, dirs, files in os.walk(_dir):
        for filename in files:
            if filename.endswith("PARAVIEW.xml"):
                filepath = os.path.join(subdir, filename)
                print("Find filepath: ", filepath)  # debug
                ImportPresets(filename=filepath)
                print("Filepath imported: ", filepath)


def main():
    '''
    main function of this module
    '''
    SCM_dir = "/home/lochy/Desktop/ScientificColourMaps7" # change before usage
    assert(os.path.isdir(SCM_dir))
    import_xmls(SCM_dir)

main()