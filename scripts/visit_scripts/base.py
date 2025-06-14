# class for working with visit
# This part contains a class (VISIT_PLOT) of basic operations
# e.g. open file vtk files, add plots, choose color schemes, etc
# Usage of this class is to be combined with classes defined for
# individual visualizations
import os
_dir = os.path.dirname(os.path.abspath(__file__))
class VISIT_PLOT():
    """
    BASE CLASS for visit plot()
    Attributes:
        idx(dict): a dictionary for index
        all_idxs(int): total indexes
        time_snap(int): current time snap number
        output_dir(str): path of output
    """

    def __init__(self, filein, **kwargs):
        """
        initiate
        Args:
            filein(str): path of vtu file
            plots(list): lists of plot
        """
        # open data base
        self.filein = filein
        OpenDatabase("localhost:%s" % filein, 0)

        # assign initial values 
        self.idxs = {}
        self.all_idxs = 0
        self.time_snap = 0
        self.output_dir = kwargs.get('output_dir', '.')

        # add plots
        # self.set_plots() is to be reloaded in children
        plot_types, vars_ = self.get_plots()
        for i in range(len(plot_types)):
            plot_type = plot_types[i]
            var_ = vars_[i]
            self.add_plot(plot_type, var_)

    def add_plot(self, plot_type, var):
        """
        add a plot
        Args:
            plot_type(str): type of plot
            var(str): variable to plot
        """
        AddPlot(plot_type, var, 1, 1)
        self.idxs[var] = self.all_idxs
        self.all_idxs += 1
        
    def draw_all(self):    
    # draw plots
        all_idxs_tuple = tuple([i for i in range(self.all_idxs)])
        SetActivePlots(all_idxs_tuple)
        DrawPlots()
        HideActivePlots()

    def goto_time_snap(self, time_snap):
        """
        goto time_snap
        Args:
            time_snap(int): number of time snap
        """
        for _ in range(time_snap - self.time_snap):
            TimeSliderNextState()
        self.time_snap = time_snap

    def __call__(self, time_snap):
        """
        call
        Args:
            time_snap(int): time_snap to plot
        """
        # shift to specific time snap
        self.goto_time_snap(time_snap)
        self.plot_time_snap()

    def plot_time_snap(self):
        """
        plot a single time step
        to be reload in children
        """
        pass

    def get_plots(self):
        """
        get types of plot
        to be reloaded in children
        Returns:
            plot_types(list)
            vars_(list): variables to plot
        """
        plot_types = []
        vars_ = []
        return plot_types, vars_
    
    def set_view_attrs(self, window_coords):
        '''
        set the attributes of camera
        '''
        # assertion
        assert len(window_coords) == 4
        # fix camera
        # Begin spontaneous state
        View2DAtts = View2DAttributes()
        View2DAtts.windowCoords = window_coords
        View2DAtts.viewportCoords = (0.2, 0.95, 0.15, 0.95)
        View2DAtts.fullFrameActivationMode = View2DAtts.Auto  # On, Off, Auto
        View2DAtts.fullFrameAutoThreshold = 100
        View2DAtts.xScale = View2DAtts.LINEAR  # LINEAR, LOG
        View2DAtts.yScale = View2DAtts.LINEAR  # LINEAR, LOG
        View2DAtts.windowValid = 1
        SetView2D(View2DAtts)
        # End spontaneous state

    def set_view_attrs_3d(self, view_normal, view_up):
        '''
        set view (camera) in 3d
        '''
        # assertion
        assert(len(view_normal) == 3)
        assert(len(view_up) == 3)
        # Begin spontaneous state
        View3DAtts = View3DAttributes()
        View3DAtts.viewNormal = view_normal
        View3DAtts.focus = (3.3915e+06, 50000, 1.445e+06)
        View3DAtts.viewUp = view_up
        View3DAtts.viewAngle = 30
        View3DAtts.parallelScale = 3.68684e+06
        View3DAtts.nearPlane = -7.37368e+06
        View3DAtts.farPlane = 7.37368e+06
        View3DAtts.imagePan = (0, 0)
        View3DAtts.imageZoom = 1
        View3DAtts.perspective = 1
        View3DAtts.eyeAngle = 2
        View3DAtts.centerOfRotationSet = 0
        View3DAtts.centerOfRotation = (3.3915e+06, 50000, 1.445e+06)
        View3DAtts.axis3DScaleFlag = 0
        View3DAtts.axis3DScales = (1, 1, 1)
        View3DAtts.shear = (0, 0, 1)
        View3DAtts.windowValid = 1
        SetView3D(View3DAtts)
        # End spontaneous state
            
    def set_rotation_transform(self, rot_deg):
        '''
        set the rotation
        '''
        AddOperator("Transform", 1)
        TransformAtts = TransformAttributes()
        TransformAtts.doRotate = 1
        TransformAtts.rotateOrigin = (0, 0, 0)
        TransformAtts.rotateAxis = (0, 0, 1)
        TransformAtts.rotateAmount = rot_deg
        TransformAtts.rotateType = TransformAtts.Deg  # Deg, Rad
        TransformAtts.inputCoordSys = TransformAtts.Cartesian  # Cartesian, Cylindrical, Spherical
        TransformAtts.outputCoordSys = TransformAtts.Cartesian  # Cartesian, Cylindrical, Spherical
        SetOperatorOptions(TransformAtts, 0, 1)
        pass

    def set_threshold(self, name_, values):
        '''
        set values of threshhold
        Inputs:
            name_(str): name of the field
            values(list of 2): lower and upper limit
        '''
        SetActivePlots((self.idxs[name_]))
        ThresholdAtts = ThresholdAttributes()
        ThresholdAtts.outputMeshType = 0
        ThresholdAtts.boundsInputType = 0
        ThresholdAtts.listedVarNames = ("default")
        ThresholdAtts.zonePortions = (1)
        ThresholdAtts.lowerBounds = (values[0])
        ThresholdAtts.upperBounds = (values[1])
        ThresholdAtts.defaultVarName = name_
        ThresholdAtts.defaultVarIsScalar = 1
        ThresholdAtts.boundsRange = ("%.6e:%.6e" % (values[0], values[1]))
        SetOperatorOptions(ThresholdAtts, 1, 0)
        DrawPlots()
    
    def set_pseudo_color(self, name_, **kwargs):
        '''
        set pseudocolor plot
        '''
        SetActivePlots(self.idxs[name_])
        PseudocolorAtts = PseudocolorAttributes()
        PseudocolorAtts.limitsMode = PseudocolorAtts.OriginalData  # OriginalData, CurrentPlot

        # color
        color_table = kwargs.get('color_table', "viridis")
        PseudocolorAtts.colorTableName = color_table

        # min and max values to use
        limits = kwargs.get('limits', None)
        if limits is None:
            pass
        else:
            assert len(limits) == 2
            PseudocolorAtts.minFlag = 1
            PseudocolorAtts.min = limits[0]
            PseudocolorAtts.useBelowMinColor = 0
            PseudocolorAtts.belowMinColor = (0, 0, 0, 255)
            PseudocolorAtts.maxFlag = 1
            PseudocolorAtts.max = limits[1]
            PseudocolorAtts.useAboveMaxColor = 0
            PseudocolorAtts.aboveMaxColor = (0, 0, 0, 255)
        
        # invert color table
        invert_color = kwargs.get('invert_color', False)
        if invert_color: 
            PseudocolorAtts.invertColorTable = 1

        # use log value
        log = kwargs.get('log', False)
        if log:
            PseudocolorAtts.scaling = PseudocolorAtts.Log  # Linear, Log, Skew
        
        # commit settings 
        SetPlotOptions(PseudocolorAtts)
    
    def plot_temperature_base(self, **kwargs):
        '''
        plot temperature
        '''
        limits = kwargs.get('limit', None)
        # set up temperature
        self.set_pseudo_color('T', color_table='SCM_vik', limits=limits)
        SetActivePlots((self.idxs['T']))
        HideActivePlots()
        # save plot 
        self.save_window('temperature')
        HideActivePlots()

    def plot_density_base(self, **kwargs):
        '''
        plot density
        '''
        limits = kwargs.get('limit', None)
        # set up density
        self.set_pseudo_color('density', color_table='magma', limits=limits)
        SetActivePlots((self.idxs['density']))
        HideActivePlots()
        # save plot 
        self.save_window('density')
        HideActivePlots()
    
    def plot_viscosity_base(self, **kwargs):
        '''
        plot viscosity
        '''
        limits = kwargs.get('limit', None)
        # set up viscosity
        self.set_pseudo_color('viscosity', color_table="SCM_roma", invert_color=False, log=True, limits=limits)
        SetActivePlots((self.idxs['viscosity']))
        HideActivePlots()
        # save plot 
        self.save_window('viscosity')
        HideActivePlots()

        
    def save_window(self, filename, **kwargs):
        '''
        save window
        Inputs:)
            filename(str): base name for file
            kwargs(dict):
                size(list of 2): size of plot in (width, height)
        '''
        SaveWindowAtts = SaveWindowAttributes()
        SaveWindowAtts.outputDirectory = self.output_dir
        SaveWindowAtts.family = 0  # we don't use family name here
        SaveWindowAtts.fileName = "%s/%s%06d" % (self.output_dir, filename, self.time_snap)
        SaveWindowAtts.format = SaveWindowAtts.PNG

        # size
        try:
            size = kwargs['size']
        except KeyError:
            pass
        else:
            assert len(size) == 2
            SaveWindowAtts.width = size[0]
            SaveWindowAtts.height = size[1]
        
        SetSaveWindowAttributes(SaveWindowAtts)
        SaveWindow()
    
    def abort(self):
        '''
        abort mission and leave
        '''
        # delete active plots
        all_idxs_tuple = tuple([i for i in range(self.all_idxs)])
        SetActivePlots(all_idxs_tuple)
        DeleteActivePlots()

        # close file 
        OpenDatabase(self.filein)