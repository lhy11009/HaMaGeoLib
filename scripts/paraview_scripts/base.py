# class for working with paraview
# This part contains a class (PARAVIEW_PLOT) of basic operations
# e.g. open file vtk files, add plots, choose color schemes, etc
# Usage of this class is to be combined with classes defined for
# individual visualizations
# use environmental variables:
# PV_INTERFACE_PATH - path to load xml files
import os
import math
# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

class PARAVIEW_PLOT():

    def __init__(self, filein, **kwargs):
        """
        initiate
        Args:
            filein(str): path of vtu file
            plots(list): lists of plot
            kwargs:
                project: name of the project
        """
        # import xmls
        PV_INTERFACE_PATH = os.environ["PV_INTERFACE_PATH"]
        if PV_INTERFACE_PATH != "" and os.path.isdir(PV_INTERFACE_PATH):
            print("import files under: %s" % PV_INTERFACE_PATH)
            import_xmls(PV_INTERFACE_PATH)

        # get variables
        project = kwargs.get("project", "ThDSubduction")
        all_variables = kwargs.get("all_variables", None)
        assert(project in ["TwoDSubduction", "ThDSubduction"])

        # make the directory to output to
        # By default, we are outputing the the "img/pv_outputs" directory
        self.output_dir = kwargs.get('output_dir', ".")
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.pv_output_dir = os.path.join(self.output_dir, "pv_outputs")
        if not os.path.isdir(self.pv_output_dir):
            os.mkdir(self.pv_output_dir)

        # all the variables we want to plot, one in these:
        # 'velocity', 'p', 'T', 'sp_upper', 'sp_lower', 'density', 'viscosity', 
        # 'current_cohesions', 'current_friction_angles', 'plastic_yielding', 'dislocation_viscosity', 
        # 'diffusion_viscosity', 'peierls_viscosity', 'error_indicator']
        # This option is either parsed in through the interface or assigned by the type of the project.
        self.view_solution_pvd = True  # if the solutionpvd is included as a view
        if all_variables is None:
            if project == "ThDSubduction":
                self.all_variables = ['velocity', 'p', 'T',  'density', 'viscosity', 'sp_upper', 'sp_lower', "strain_rate"]
                HAS_PLATE_EDGE = True
                if HAS_PLATE_EDGE:
                    self.all_variables.append('plate_edge')
            elif project == "TwoDSubduction":
                self.all_variables = ['velocity', 'p', 'T',  'density', 'viscosity', 'spcrust', 'spharz',\
                'dislocation_viscosity', 'diffusion_viscosity', 'peierls_viscosity', 'strain_rate', 'nonadiabatic_pressure']
        else:
            assert(type(all_variables) == list)
            for i in all_variables:
                assert(type(i) == str)
            self.all_variables = all_variables

        # open data base
        # the "solution.pvd" file is read in.
        # This is a file linked to all the "solution_*.pvtu files for different output steps,
        # which in turn leads to all the individual .vtu files at the step.
        self.filein = filein
        # assign initial values 
        self.idxs = {}
        self.all_idxs = 0
        self.time = 0.0
        self.solutionpvd = PVDReader(registrationName='solution.pvd', FileName=filein)
        #  allow us to select from the fields in the pvd file (list them in the all_variables array)
        self.solutionpvd.PointArrays = self.all_variables
        self.camera_dict = {}  # a dictionary for saving camera settings
        self.colorbar_dict = {} # a dictionary for saving colorbar settings
        # get animation scene
        animationScene1 = GetAnimationScene()
        # update animation scene based on data timesteps
        animationScene1.UpdateAnimationUsingDataTimeSteps()
        self.time = animationScene1.AnimationTime
        # get active view
        self.renderView1 = GetActiveViewOrCreate('RenderView')
        if self.view_solution_pvd:
            # show data in view
            solutionpvdDisplay = Show(self.solutionpvd, self.renderView1, 'UnstructuredGridRepresentation')
            # hide data in view
            Hide(self.solutionpvd, self.renderView1)
        # update the view to ensure updated data information
        self.renderView1.Update()

    def goto_time(self, _time):
        '''
        go to a different snapshot
        Inputs:
            _time: time assigned
        '''
        animationScene1 = GetAnimationScene()
        animationScene1.AnimationTime = _time
        self.time = _time
        print("Time: %.4e" % _time)


def import_xmls(_dir, **kwargs):
    '''
    Import xmls files
    Inputs:
        _dir: directory contains xml files
        **kwargs:
            debug - output debug options
    '''
    debug = kwargs.get("debug", False)
    i = 0
    for subdir, dirs, files in os.walk(_dir):
        for filename in files:
            if filename.endswith("PARAVIEW.xml"):
                filepath = os.path.join(subdir, filename)
                if debug:
                    print("Find filepath: ", filepath)  # debug
                ImportPresets(filename=filepath)
                i += 1
                if debug:
                    print("Filepath imported: ", filepath)
    print("%d file imported" % i)


def apply_rotation(_source, origin, angles, **kwargs):
    '''
    apply rotation to a dataset
    Inputs:
        source(str): _source of data
        origin (list): the origin point of visualization
        angles (list): the angles of rotation in x, y, z
        kwargs:
            registrationName : the name of registration
    '''
    registrationName = kwargs.get("registrationName", 'Transform')

    # find source
    solutionpvd = FindSource(_source)

    # create a new 'Transform'
    transform = Transform(registrationName=registrationName, Input=solutionpvd)
    transform.Transform = 'Transform'

    # Properties modified on transform1.Transform
    transform.Transform.Translate = origin
    # Properties modified on transform1.Transform
    transform.Transform.Rotate = angles

    # same as uncheck "show box"
    Hide3DWidgets()


# Note: to add a feature in script, it follows the same structure
# by first calling the class and create an object and the twik the attribute
# of that object.
def add_slice(solutionpvd, field, Origin, Normal, **kwargs):
    '''
    create a new 'Slice'
    Inputs:
        solutionpvd: a solutionpvd object of paraview
        field: field to show
        Origin: the origin point of visualization
        Normal: the normal direction of the slice
    '''
    # get additional variables
    renderView0 = kwargs.get('renderView', None)
    _name = kwargs.get('name', '0')
    # create slice
    slice1 = Slice(registrationName='slice_%s' % _name, Input=solutionpvd)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    slice1.SliceOffsetValues = [0.0]
    # init the 'Plane' selected for 'SliceType'
    slice1.SliceType.Origin = Origin
    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice1.HyperTreeGridSlicer.Origin = [2000000.0, 2000000.0, 500000.0]
    # Modify the normal direction
    slice1.SliceType.Normal = Normal
    # get active view
    if renderView0 == None:
        renderView1 = GetActiveViewOrCreate('RenderView')
    else:
        renderView1 = renderView0
    # show data in view
    slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')
    adjust_slice_color(slice1Display, field, renderView1)
    renderView1.Update()
    return slice1, slice1Display, renderView1


def add_clip(solutionpvd, Origin, Normal, **kwargs):
    '''
    create a new 'Clip'
    Inputs:
        solutionpvd: a solutionpvd object of paraview
        field: field to show
        Origin: the origin point of visualization
        Normal: the normal direction of the slice
    '''
    # get additional variables
    renderView0 = kwargs.get('renderView', None)
    _name = kwargs.get('name', '0')
    clip1 = Clip(registrationName="clip_%s" % _name, Input=solutionpvd)
    clip1.ClipType = 'Plane'
    clip1.HyperTreeGridClipper = 'Plane'
    # clip1.Scalars = ['POINTS', 'T']
    # clip1.Value = 1136.5
    # init the 'Plane' selected for 'ClipType'
    clip1.ClipType.Origin = Origin
    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip1.HyperTreeGridClipper.Origin = Origin
    # Properties modified on clip1.ClipType
    clip1.ClipType.Normal = Normal
    if renderView0 == None:
        renderView1 = GetActiveViewOrCreate('RenderView')
    else:
        renderView1 = renderView0
    # show data in view
    clip1Display = Show(clip1, renderView1, 'GeometryRepresentation')
    renderView1.Update()
    return clip1, clip1Display, renderView1



def add_isovolume(solutionpvd, field, thresholds, **kwargs):
    '''
    Inputs:
        solutionpvd: a solutionpvd object of paraview
        field: field to show
        thresholds: thresholds to apply
    '''
    renderView0 = kwargs.get('renderView', None)
    _name = kwargs.get('name', 0)
    assert(len(thresholds) == 2)
    isoVolume1 = IsoVolume(registrationName='isoVolume_%s' % _name, Input=solutionpvd)
    isoVolume1.InputScalars = ['POINTS', field]
    isoVolume1.ThresholdRange = thresholds
    # get active view
    if renderView0 == None:
        renderView1 = GetActiveViewOrCreate('RenderView')
    else:
        renderView1 = renderView0
    # show data in view
    isoVolume1Display = Show(isoVolume1, renderView1, 'GeometryRepresentation')
    renderView1.Update()
    return isoVolume1, isoVolume1Display, renderView1


def add_glyph(_source, field, scalefactor, nsample, **kwargs):
    '''
    Inputs:
        solutionpvd: a solutionpvd object of paraview
        field: field to show
        scalefactor: a factor for the scaling of the lenght of the field
        nsample: number of sample points
    '''
    # addtional variables
    registrationName = kwargs.get("registrationName", 'Glyph1')
    OrientationArray = kwargs.get("OrientationArray", None)
    ScaleArray = kwargs.get("ScaleArray", None)
    GlyphType = kwargs.get("GlyphType", "Arrow")
    
    # get active source and renderview
    pvd = FindSource(_source)
    renderView1 = GetActiveViewOrCreate('RenderView')
    
    glyph1 = Glyph(registrationName=registrationName, Input=pvd, GlyphType=GlyphType)
    if OrientationArray is not None:
        glyph1.OrientationArray = ['POINTS', OrientationArray]
    else:
        glyph1.OrientationArray = ['POINTS', 'No orientation array']
    if ScaleArray is not None:
        glyph1.ScaleArray = ['POINTS', ScaleArray]
    else:
        glyph1.ScaleArray = ['POINTS', 'No scale array']
    glyph1.ScaleFactor = scalefactor
    glyph1.GlyphTransform = 'Transform2'
    glyph1.MaximumNumberOfSamplePoints = nsample
    

    glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
    # set the vector line width
    glyph1Display.LineWidth = 2.0
    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(renderView1, True)
    # get color transfer function/color map for 'field'
    fieldLUT = GetColorTransferFunction(field)
    # get opacity transfer function/opacity map for 'field'
    fieldPWF = GetOpacityTransferFunction(field)

    # change the color scheme 
    fieldLUT.ApplyPreset('X Ray', True)

    # set scalar coloring
    ColorBy(glyph1Display, ('POINTS', field, 'Magnitude'))
    
    # hide glaph in view
    Hide(glyph1, renderView1)
    
    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)
    
    # update the view to ensure updated data information
    renderView1.Update()


    
def adjust_slice_color(slice1Display, field, renderView):
    '''
    adjust the color scheme of a slice filter
    Inputs:
        slice1Display : a display object we created before this
        field : field to show
        renderView: a renderView object from paraview
    '''
    # set scalar coloring
    ColorBy(slice1Display, ('POINTS', field))
    # Hide the scalar bar for this color map if no visible data is colored by it.
    # HideScalarBarIfNotNeeded(vtkBlockColorsLUT, renderView1)
    # rescale color and/or opacity maps used to include current data range
    slice1Display.RescaleTransferFunctionToDataRange(True, False)
    # show color bar/color legend
    slice1Display.SetScalarBarVisibility(renderView, True)
    # get color transfer function/color map for field
    sp_upperLUT = GetColorTransferFunction(field)
    # get opacity transfer function/opacity map for field
    sp_upperPWF = GetOpacityTransferFunction(field)


def rescale_transfer_function_combined(field, min_val, max_val):

    # get color transfer function/color map for 'T'
    # Rescale transfer function
    tLUT = GetColorTransferFunction(field)
    tLUT.RescaleTransferFunction(min_val, max_val)

    # get opacity transfer function/opacity map for 'T'
    # Rescale transfer function
    tPWF = GetOpacityTransferFunction(field)
    tPWF.RescaleTransferFunction(min_val, max_val)

    # get 2D transfer function for 'T'
    # Rescale 2D transfer function
    try:
        tTF2D = GetTransferFunction2D(field)
    except NameError:
        pass
    else:
        tTF2D.RescaleTransferFunction(min_val, max_val, 0.0, 1.0)


def adjust_camera(renderView, position, focalPoint, parallelScale, viewUp):
    '''
    adjust camera
    Inputs:
        renderView: a "renderView" object of Paraview
    '''
    # reset view to fit data
    renderView.ResetCamera(False)
    # current camera placement for renderView
    renderView.InteractionMode = '2D'
    renderView.CameraPosition = position
    renderView.CameraFocalPoint = focalPoint
    renderView.CameraParallelScale = parallelScale
    renderView.CameraViewUp = viewUp


def adjust_color_bar(colorTransferFunc, renderView, color_bar_config):
    '''
    adjust color bar
    '''
    assert(len(color_bar_config) == 2)
    position = color_bar_config[0]
    length = color_bar_config[1]
    assert(len(position) == 2)
    # get color legend/bar for sp_upperLUT in view renderView1
    colorBar = GetScalarBar(colorTransferFunc, renderView)
    # change scalar bar placement
    colorBar.Position = position
    colorBar.ScalarBarLength = length


def add_plot(_source, field, **kwargs):
    '''
    simply add a plot on the original viewer
    Inputs:
        _source (str): the data source
        field (str): name of the field to plot
        kwargs:
            use_log - use log value
            lim - limits of values
            color - color scheme to use
    '''
    # get inputs
    use_log = kwargs.get("use_log", False)
    lim = kwargs.get("lim", None)
    _color = kwargs.get("color", None)
    # todo_density
    invert = kwargs.get("invert", False)

    # get active source.
    pvd = FindSource(_source)
    # set active source
    SetActiveSource(pvd)
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # get display properties
    # pvdDisplay = GetDisplayProperties(pvd, view=renderView1)
    pvdDisplay = Show(pvd, renderView1, 'UnstructuredGridRepresentation')
    # set scalar coloring
    ColorBy(pvdDisplay, ('POINTS', field))
    
    
    # rescale color and/or opacity maps used to include current data range
    pvdDisplay.RescaleTransferFunctionToDataRange(True, False)
    # show color bar/color legend
    pvdDisplay.SetScalarBarVisibility(renderView1, True)
    # get color transfer function/color map for 'field'
    fieldLUT = GetColorTransferFunction(field)
    # get opacity transfer function/opacity map for 'field'
    fieldPWF = GetOpacityTransferFunction(field)

    # reset limit 
    fieldLUT.RescaleTransferFunction(lim[0], lim[1]) # Rescale transfer function
    fieldPWF.RescaleTransferFunction(lim[0], lim[1]) # Rescale transfer function
    # set using log values
    if use_log:
        # convert to log space
        fieldLUT.MapControlPointsToLogSpace()
        # Properties modified on fieldLUT
        fieldLUT.UseLogScale = 1
    # apply a color scheme
    if _color is not None: 
        # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
        fieldLUT.ApplyPreset(_color, True)
        if invert:
            fieldLUT.InvertTransferFunction()
   
    # reset view to fit data
    # without this, the figure won't show
    renderView1.ResetCamera(False)

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)
    # hide data in view
    Hide(pvd, renderView1)