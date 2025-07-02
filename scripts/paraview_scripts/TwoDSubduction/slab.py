# variable to substitute
# vts file:
#   PARAVIEW_FILE
# directory for images:
#   IMG_OUTPUT_DIR
# y coordinates for trench edge
#   TRENCH_EDGE_Y
# minimum viscosity
#   ETA_MIN
# maximum viscosity
#   ETA_MAX
# rotation of the domain
#   ROTATION_ANGLE
# geometry of the model
#   GEOMETRY
# plot model axis
#   PLOT_AXIS
# types of plot (a list, e.g. upper mante / whole)
#   PLOT_TYPES
# additional fields to parse
#   ADDITIONAL_FIELDS
# outer radius
#   OUTER_RADIUS

# todo_cv
class SLAB(PARAVIEW_PLOT):
    '''
    Inherit frome PARAVIEW_PLOT
    Usage is for plotting the shape of the 3D slabs
    '''
    def __init__(self, filein, **kwargs):
        '''
        Initiation
        '''
        # First, call the initiation function of the base function
        # The "project" entry does nothing other than tag the class.
        # The 'all_variables' makes use of the all_variables entry in the base class's initiation.
        kwargs['project'] = "TwoDSubduction"
        kwargs['all_variables'] = ['velocity', 'p', 'T',  'density', 'viscosity', 'spcrust', 'spharz',\
                'dislocation_viscosity', 'diffusion_viscosity', 'peierls_viscosity', 'strain_rate', 'nonadiabatic_pressure']\
                    +  ADDITIONAL_FIELDS
       
        PARAVIEW_PLOT.__init__(self, filein, **kwargs)
        
        # load the types of plots
        plot_types = PLOT_TYPES

        self.eta_min = ETA_MIN
        self.eta_max = ETA_MAX
        self.T_min = 273.0
        self.T_max = 2273.0
        # todo_density
        self.density_min = 3000.0
        self.density_max = 4000.0
        self.camera_dict['twod_upper_mantle'] = [[0.0, 6e6, 2.5e7],[0.0, 6e6, 0.0], 5.4e5, None]
        # todo_Vres
        add_velocity_residual("solution.pvd", registrationName="pFilter_Vres")
        
        # add rotation 
        apply_rotation("solution.pvd", [0.0, 0.0, 0.0], [0.0, 0.0, ROTATION_ANGLE], registrationName="Transform1")
        add_plot("Transform1", "viscosity", use_log=True, lim=[self.eta_min, self.eta_max], color="bilbao")
        add_plot("Transform1", "T", lim=[self.T_min, self.T_max], color="lapaz")
        add_plot("Transform1", "density", lim=[self.density_min, self.density_max], color="batlow")

        if INCLUDE_METASTABLE:
            add_plot("Transform1", "metastable", lim=[0.0, 1.0], color="Viridis (matplotlib)")

        # add glyph
        add_glyph1("Transform1", "velocity", 1e6, registrationName="Glyph1")

        # add plot of composition
        add_plot("Transform1", "spcrust", lim=[0.0, 1.0], color="tokyo")

        # add plot of deformation mechanism
        add_deformation_mechanism("Transform1", registrationName="pFilter_DM")


        # Extract points
        # Extract a selection of points, one with smaller longitude, one with bigger longitude
        # Then set color to "solid"
        # Note: For now, I can only get one point to work
        if False:
            _source="Transform1" # name
            pvd = FindSource(_source) # pvd - the actual source
            SetActiveSource(pvd)
            renderView1 = GetActiveViewOrCreate('RenderView')

            first_lon = 20.0 # first point
            x1, y1 ,z1 = ggr2cart(0.0, (first_lon + ROTATION_ANGLE) / 180.0 * 3.1415926, OUTER_RADIUS)
            # position = [1965877.371, 6060112.801, 0.0] # first point
            distance = 10e3 
            QuerySelect(QueryString='(pointIsNear([(%.4f, %.4f, %.4f),], %.4f, inputs))' % (x1, y1, z1, distance),\
                FieldType='POINT', InsideOut=0)
        
            extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=pvd)
        
            extractSelection1Display = Show(extractSelection1, renderView1, 'UnstructuredGridRepresentation')
            field0 = extractSelection1Display.ColorArrayName[1]
            field0LUT = GetColorTransferFunction(field0)
        
            ColorBy(extractSelection1Display, None)
            extractSelection1Display.AmbientColor = [0.6666666666666666, 0.0, 1.0]
            extractSelection1Display.DiffuseColor = [0.6666666666666666, 0.0, 1.0]
            extractSelection1Display.PointSize = 2.0

            HideScalarBarIfNotNeeded(field0LUT, renderView1)
        
            
            second_lon = 45.0 # second point
            x2, y2 ,z2 = ggr2cart(0.0, (second_lon + ROTATION_ANGLE) / 180.0 * 3.1415926, OUTER_RADIUS)
            # position = [1965877.371, 6060112.801, 0.0] # first point
            distance = 10e3 
            QuerySelect(QueryString='(pointIsNear([(%.4f, %.4f, %.4f),], %.4f, inputs))' % (x2, y2, z2, distance),\
                FieldType='POINT', InsideOut=0)
        
            extractSelection2 = ExtractSelection(registrationName='ExtractSelection2', Input=pvd)
        
            extractSelection2Display = Show(extractSelection2, renderView1, 'UnstructuredGridRepresentation')
            field0 = extractSelection2Display.ColorArrayName[1]
            field0LUT = GetColorTransferFunction(field0)
        
            ColorBy(extractSelection2Display, None)
            extractSelection2Display.AmbientColor = [0.6666666666666666, 0.0, 1.0]
            extractSelection2Display.DiffuseColor = [0.6666666666666666, 0.0, 1.0]
            extractSelection2Display.PointSize = 2.0

            HideScalarBarIfNotNeeded(field0LUT, renderView1)
        
            Hide(pvd, renderView1)
            Hide(extractSelection1, renderView1)
            Hide(extractSelection2, renderView1)

        # add contour T
        renderView1 = GetActiveViewOrCreate('RenderView')

        _source="Transform1" # name
        pvd = FindSource(_source) # pvd - the actual source
        fieldTLUT = GetColorTransferFunction('T')

        # add contour of T
        contourT1 = Contour(registrationName='ContourT1', Input=pvd)
        contourT1.ContourBy = ['POINTS', 'T']
        if "upper_mantle" in plot_types or "wedge" in plot_types:
            contourT1.Isosurfaces = [673.15, 1073.15]
        elif "wedge_small" in plot_types:
            contourT1.Isosurfaces = [673.15, 873.15, 1073.15, 1273.15]
        elif "wedge_bigger" or "wedge_02252025" in plot_types:
            contourT1.Isosurfaces = [373.15, 473.15, 573.15, 673.15, 773.15, 873,15, 973.15, 1073.15, 1173.15, 1273.15]
        else:
            contourT1.Isosurfaces = [1373.15]

        contourT1Display = Show(contourT1, renderView1, 'GeometryRepresentation')
        contourT1Display.LineWidth = 4.0
        contourT1Display.Ambient = 1.0

        # add contour of sp_crust
        contourCr = Contour(registrationName='ContourCr', Input=pvd)
        contourCr.ContourBy = ['POINTS', 'spcrust']
        contourCr.Isosurfaces = [0.8]
        contourCrDisplay = Show(contourCr, renderView1, 'GeometryRepresentation')
        contourCrDisplay.LineWidth = 2.0
        contourCrDisplay.AmbientColor = [0.3333333333333333, 0.3333333333333333, 0.0]
        contourCrDisplay.ColorArrayName = [None, '']
        contourCrDisplay.DiffuseColor = [0.3333333333333333, 0.3333333333333333, 0.0]

        # hide all plots
        Hide(contourT1, renderView1)
        Hide(contourCr, renderView1)
        HideScalarBarIfNotNeeded(fieldTLUT, renderView1)


    def plot_step_upper_mantle(self, **kwargs): 
        """
        Plot a step in the upper mantle with visualizations including scalar and vector fields.
        
        Parameters:
        - kwargs (dict): 
            - glyphRegistrationName (str): Name of the glyph, used to adjust the glyph outputs.
        """
        # Parse input for glyph registration name, defaulting to "Glyph1".
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # Get the active view and configure background color and settings.
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # Define source and field parameters for the plot.
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"  # Scalar field for temperature.
        field2 = "viscosity"  # Scalar field for viscosity.
        field3 = "spcrust"  # Scalar field for crustal properties.
        layout_resolution = (1350, 704)

        # Retrieve and configure color transfer functions for the fields.
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)

        # Find the sources for scalar and vector fields.
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)

        # Display the scalar field (source1) in the render view and configure settings.
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            # Display axes grid and configure axis colors.
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

        # Configure temperature field settings for the scalar plot.
        field1LUT = GetColorTransferFunction(field1)
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)
        rescale_transfer_function_combined('T', 273.0, 1673.0)
            

        # Configure the color bar for the temperature field.
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'

        # Hide the orientation axes.
        renderView1.OrientationAxesVisibility = 0

        # Display the vector field (sourceV) and configure its color transfer function.
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
        ColorBy(sourceVDisplay, None)
        fieldVLUT = GetColorTransferFunction('velocity')
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)

        # Configure the color bar for the velocity field.
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'

        # Adjust the position of the point source and show related annotations.
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        _source_v_re = _source_v + "_representative"
        sourceVRE = FindSource(_source_v_re)
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation')
        _source_v_txt = _source_v + "_text"
        sourceVTXT = FindSource(_source_v_txt)
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]

        # Adjust glyph properties based on the specified parameters.
        scale_factor = 1e6
        n_sample_points = 20000
        point_source_center = [0.0, 0.0, 0.0]
        if "GEOMETRY" == "chunk":
            point_source_center = [0, 6.4e6, 0]
        elif "GEOMETRY" == "box":
            point_source_center = [4.65e6, BOX_THICKNESS + 60e3, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)

        # Show contour
        fieldTLUT = GetColorTransferFunction("T")
        source_contour = FindSource("ContourT1")

        contourTDisplay = Show(source_contour, renderView1, 'GeometryRepresentation')
        
        rescale_transfer_function_combined('T', 273.0, 1673.0)

        # Configure layout and camera settings based on geometry.
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [-74708.2999944719, 5867664.065060813, 24790239.31741349]
            renderView1.CameraFocalPoint = [-74708.2999944719, 5867664.065060813, 0.0]
            renderView1.CameraParallelScale = 651407.1273990012
        elif "GEOMETRY" == "box":
            renderView1.CameraPosition = [4700895.868280185, BOX_THICKNESS-351083.410241, 15340954.822755022]
            renderView1.CameraFocalPoint = [4700895.868280185, BOX_THICKNESS-351083.410241, 0.0]
            renderView1.CameraParallelScale = 487763.78047352127
        
        # Simply comments all the following to debug

        # Save the first figure (temperature field).
        fig_path = os.path.join(self.pv_output_dir, "T_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # Plot the second scalar field (viscosity) and configure settings.
        field2 = "viscosity"
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        field2LUT = GetColorTransferFunction(field2)
        source1Display.SetScalarBarVisibility(renderView1, True)
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        rescale_transfer_function_combined('viscosity', ETA_MIN, ETA_MAX)

        # Save the second figure (viscosity field).
        fig_path = os.path.join(self.pv_output_dir, "viscosity_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # todo_density
        # Plot the third scalar field (viscosity) and configure settings.
        field3 = "density"
        ColorBy(source1Display, ('POINTS', field3, 'Magnitude'))
        field3LUT = GetColorTransferFunction(field3)
        source1Display.SetScalarBarVisibility(renderView1, True)
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        rescale_transfer_function_combined(field3, self.density_min, self.density_max)

        # Save the second figure (viscosity field).
        fig_path = os.path.join(self.pv_output_dir, "density_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "density_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        if INCLUDE_METASTABLE:
            field4 = "metastable"
            ColorBy(source1Display, ('POINTS', field4, 'Magnitude'))
            field4LUT = GetColorTransferFunction(field4)
            source1Display.SetScalarBarVisibility(renderView1, True)
            HideScalarBarIfNotNeeded(field3LUT, renderView1)
            rescale_transfer_function_combined(field4, 0.0, 1.0)

            # Save the second figure (viscosity field).
            fig_path = os.path.join(self.pv_output_dir, "metastable_t%.4e.pdf" % self.time)
            fig_png_path = os.path.join(self.pv_output_dir, "metastable_t%.4e.png" % self.time)
            SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
            ExportView(fig_path, view=renderView1)


        # Hide all the plots and scalar bars to clean up.
        # Hide(source1, renderView1)
        # Hide(sourceV, renderView1)
        # Hide(sourceVRE, renderView1)
        # Hide(sourceVTXT, renderView1)
        # Hide(source_contour, renderView1)
        # HideScalarBarIfNotNeeded(field2LUT, renderView1)
        # HideScalarBarIfNotNeeded(fieldVLUT, renderView1)
        # HideScalarBarIfNotNeeded(fieldTLUT, renderView1)


    def plot_step_upper_mantle_DM(self, **kwargs): 
        """
        Plot a step in the upper mantle with visualizations including scalar and vector fields.
        
        Parameters:
        - kwargs (dict): 
            - glyphRegistrationName (str): Name of the glyph, used to adjust the glyph outputs.
        """
        # Parse input for glyph registration name, defaulting to "Glyph1".
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # Get the active view and configure background color and settings.
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # Define source and field parameters for the plot.
        _source = "pFilter_DM"
        _source_v = "Glyph1"
        field1 = "deformation_mechanism"  # Scalar field for temperature.
        layout_resolution = (1350, 704)

        # Retrieve and configure color transfer functions for the fields.

        # Find the sources for scalar and vector fields.
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)

        # Display the scalar field (source1) in the render view and configure settings.
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            # Display axes grid and configure axis colors.
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

        # Configure temperature field settings for the scalar plot.
        field1LUT = GetColorTransferFunction(field1)
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        field1LUT.RescaleTransferFunction(0.0, 3.0)
        field1LUT.NumberOfTableValues = 4

        # Configure the color bar for the temperature field.
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'

        # Hide the orientation axes.
        renderView1.OrientationAxesVisibility = 0

        # Display the vector field (sourceV) and configure its color transfer function.
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
        fieldVLUT = GetColorTransferFunction('velocity')
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)

        # Configure the color bar for the velocity field.
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'

        # Adjust the position of the point source and show related annotations.
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        _source_v_re = _source_v + "_representative"
        sourceVRE = FindSource(_source_v_re)
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation')
        _source_v_txt = _source_v + "_text"
        sourceVTXT = FindSource(_source_v_txt)
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]

        # Adjust glyph properties based on the specified parameters.
        scale_factor = 1e6
        n_sample_points = 20000
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [0, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)

        # Configure layout and camera settings based on geometry.
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [-22000.0, 334282.0428107196, 24950883.772515625]
            renderView1.CameraFocalPoint = [-22000.0, 5774282.042810716, -49116.227484387426]
            renderView1.CameraViewUp = [0.0, 0.9771340138064095, 0.2126243614042745]
            renderView1.CameraParallelScale = 968000.0000000001
        elif "GEOMETRY" == "box":
            renderView1.CameraPosition = [4700895.868280185, 2538916.5897593317, 15340954.822755022]
            renderView1.CameraFocalPoint = [4700895.868280185, 2538916.5897593317, 0.0]
            renderView1.CameraParallelScale = 487763.78047352127
        
        # Save the deformation mechanism plot.
        fig_path = os.path.join(self.pv_output_dir, "dm_t%.4e.eps" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "dm_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)
        
        # # Hide all the plots and scalar bars to clean up.
        Hide(source1, renderView1)
        Hide(sourceV, renderView1)
        Hide(sourceVRE, renderView1)
        Hide(sourceVTXT, renderView1)
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        HideScalarBarIfNotNeeded(fieldVLUT, renderView1)

    
    def plot_step_whole(self, **kwargs): 
        '''
        plot a step
        Inputs:
            kwargs:
                glyphRegistrationName: the name of the glyph, used to adjust the glyph outputs
        '''
        # parse input
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # get active view and source
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # set source and field
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"
        field2 = "viscosity"
        field3 = "spcrust"
        layout_resolution = (1350, 704)
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)
       
        # find source
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)
    
        # show source1
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        # get color transfer function/color map for 'field'
        field1LUT = GetColorTransferFunction(field1)
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)
        # Rescale transfer function, 2d transfer function
        field1LUT.RescaleTransferFunction(273.0, 2273.0)
        # colorbar position
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        
        # show sourceV (vector field)
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
        # get color transfer function/color map for 'field'
        fieldVLUT = GetColorTransferFunction('velocity')
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)
        # colorbar position
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0


        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
        
        scale_factor = 1e6
        n_sample_points = 20000
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [0, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError() 
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [1e5, 4.8e6, 2.5e7]
            renderView1.CameraFocalPoint = [1e5, 4.8e6, 0.0]
            renderView1.CameraParallelScale = 2.1e6
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "T_whole_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_whole_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # second plot
        field2 = "viscosity"
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [1e5, 4.8e6, 2.5e7]
            renderView1.CameraFocalPoint = [1e5, 4.8e6, 0.0]
            renderView1.CameraParallelScale = 2.1e6
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field2LUTColorBar = GetScalarBar(field2LUT, renderView1)
        field2LUTColorBar.Orientation = 'Horizontal'
        field2LUTColorBar.WindowLocation = 'Any Location'
        field2LUTColorBar.Position = [0.041, 0.908]
        field2LUTColorBar.ScalarBarLength = 0.33
        field2LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.TitleFontFamily = 'Times'
        field2LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field2PWF = GetOpacityTransferFunction('viscosity')
        field2LUT.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        field2PWF.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_whole_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_whole_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # hide plots
        Hide(source1, renderView1)
        Hide(sourceV, renderView1)
        Hide(sourceVRE, renderView1)
        Hide(sourceVTXT, renderView1)
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(fieldVLUT, renderView1)
    
    def plot_step_whole_whole(self, **kwargs): 
        '''
        plot a step
        Inputs:
            kwargs:
                glyphRegistrationName: the name of the glyph, used to adjust the glyph outputs
        '''
        # parse input
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # get active view and source
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # set source and field
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"
        field2 = "viscosity"
        field3 = "spcrust"
        layout_resolution = (1350, 704)
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)
       
        # find source
        source1 = FindSource(_source)
    
        # show source1
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        # get color transfer function/color map for 'field'
        field1LUT = GetColorTransferFunction(field1)
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)
        # Rescale transfer function, 2d transfer function
        field1LUT.RescaleTransferFunction(273.0, 3500.0)
        # colorbar position
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        
        # show sourceV (vector field), not included
        
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [5e4, 2.8e6, 2.5e7]
            renderView1.CameraFocalPoint = [5e4, 2.8e6, 0.0]
            renderView1.CameraParallelScale = 5e6
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "T_whole_whole_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_whole_whole_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # second plot
        field2 = "viscosity"
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [5e4, 2.8e6, 2.5e7]
            renderView1.CameraFocalPoint = [5e4, 2.8e6, 0.0]
            renderView1.CameraParallelScale = 5e6
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field2LUTColorBar = GetScalarBar(field2LUT, renderView1)
        field2LUTColorBar.Orientation = 'Horizontal'
        field2LUTColorBar.WindowLocation = 'Any Location'
        field2LUTColorBar.Position = [0.041, 0.908]
        field2LUTColorBar.ScalarBarLength = 0.33
        field2LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.TitleFontFamily = 'Times'
        field2LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field2PWF = GetOpacityTransferFunction('viscosity')
        field2LUT.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        field2PWF.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_whole_whole_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_whole_whole_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # hide plots
        Hide(source1, renderView1)
        HideScalarBarIfNotNeeded(field2LUT, renderView1)


    def plot_step_wedge_small(self, **kwargs): 
        '''
        plot a step and focus on the wedge
        Inputs:
            kwargs:
                glyphRegistrationName: the name of the glyph, used to adjust the glyph outputs
        '''
        # parse input
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # get active view and source
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # set source and field
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"
        field2 = "viscosity"
        field3 = "spcrust"
        layout_resolution = (1350, 704)
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)
       
        # find source
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)
    
        # show source1
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        
        # Configure temperature field settings for the scalar plot.
        field1LUT = GetColorTransferFunction(field1)
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)
        rescale_transfer_function_combined('T', 273.0, 1673.0)

        # colorbar position
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        
        # show sourceV (vector field)
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
        # get color transfer function/color map for 'field'
        fieldVLUT = GetColorTransferFunction('velocity')
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)
        # colorbar position
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0


        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
        # set the camera position
        CameraPosition = [-5812.506046829139, 6295461.30365194, 25000000.0]
        CameraFocalPoint = [-5812.506046829139, 6295461.30365194, 0.0]
        CameraParallelScale = 39173.24444222669

        # Set Glyph
        scale_factor = 100000
        n_sample_points = 800000
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [CameraPosition[0] - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        
        # Show contours
        fieldTLUT = GetColorTransferFunction("T")

        source_contour1 = FindSource("ContourT1")
        contourT1Display = Show(source_contour1, renderView1, 'GeometryRepresentation')
        
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScale
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "T_wedge_small_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_wedge_small_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # Comment the codes below to debug
        # second plot
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScale
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field2LUTColorBar = GetScalarBar(field2LUT, renderView1)
        field2LUTColorBar.Orientation = 'Horizontal'
        field2LUTColorBar.WindowLocation = 'Any Location'
        field2LUTColorBar.Position = [0.041, 0.908]
        field2LUTColorBar.ScalarBarLength = 0.33
        field2LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.TitleFontFamily = 'Times'
        field2LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field2PWF = GetOpacityTransferFunction('viscosity')
        field2LUT.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        field2PWF.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_wedge_small_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_wedge_small_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # third plot
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field3, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScale
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field3LUTColorBar = GetScalarBar(field3LUT, renderView1)
        field3LUTColorBar.Orientation = 'Horizontal'
        field3LUTColorBar.WindowLocation = 'Any Location'
        field3LUTColorBar.Position = [0.041, 0.908]
        field3LUTColorBar.ScalarBarLength = 0.33
        field3LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field3LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field3LUTColorBar.TitleFontFamily = 'Times'
        field3LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field3LUT = GetColorTransferFunction(field3)
        field3PWF = GetOpacityTransferFunction(field3)
        field3LUT.RescaleTransferFunction(0.0, 1.0)
        field3PWF.RescaleTransferFunction(0.0, 1.0)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "%s_wedge_small_t%.4e.pdf" % (field3, self.time))
        fig_png_path = os.path.join(self.pv_output_dir, "%s_wedge_small_t%.4e.png" % (field3, self.time))
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # hide plots
        Hide(source1, renderView1)
        Hide(sourceV, renderView1)
        Hide(sourceVRE, renderView1)
        Hide(sourceVTXT, renderView1)
        Hide(source_contour1, renderView1)
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        HideScalarBarIfNotNeeded(fieldVLUT, renderView1)
        HideScalarBarIfNotNeeded(fieldTLUT, renderView1)

    def plot_step_wedge(self, **kwargs): 
        '''
        plot a step and focus on the wedge
        Inputs:
            kwargs:
                glyphRegistrationName: the name of the glyph, used to adjust the glyph outputs
        '''
        # parse input
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # get active view and source
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # set source and field
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"
        field2 = "viscosity"
        field3 = "spcrust"
        layout_resolution = (1350, 704)
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)
       
        # find source
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)
    
        # show source1
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        
        # Configure temperature field settings for the scalar plot.
        field1LUT = GetColorTransferFunction(field1)
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)
        rescale_transfer_function_combined('T', 273.0, 1673.0)

        # colorbar position
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        
        # show sourceV (vector field)
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
        # get color transfer function/color map for 'field'
        fieldVLUT = GetColorTransferFunction('velocity')
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)
        # colorbar position
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0


        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]

        # Set Camera parameters
        camera_x = 399.56120266253004

        # Set Glyph
        scale_factor = 5e5
        n_sample_points = 100000
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [camera_x - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        
        # Show contours
        fieldTLUT = GetColorTransferFunction("T")
        source_contour1 = FindSource("ContourT1")

        contourT1Display = Show(source_contour1, renderView1, 'GeometryRepresentation')
        
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6280121.52990357, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6280121.52990357, 0.0]
            renderView1.CameraParallelScale = 122942.42196571268
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "T_wedge_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_wedge_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # Comment the codes below to debug
        # second plot
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6280121.52990357, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6280121.52990357, 0.0]
            renderView1.CameraParallelScale = 122942.42196571268
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field2LUTColorBar = GetScalarBar(field2LUT, renderView1)
        field2LUTColorBar.Orientation = 'Horizontal'
        field2LUTColorBar.WindowLocation = 'Any Location'
        field2LUTColorBar.Position = [0.041, 0.908]
        field2LUTColorBar.ScalarBarLength = 0.33
        field2LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.TitleFontFamily = 'Times'
        field2LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field2PWF = GetOpacityTransferFunction('viscosity')
        field2LUT.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        field2PWF.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_wedge_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_wedge_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # third plot
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field3, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6280121.52990357, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6280121.52990357, 0.0]
            renderView1.CameraParallelScale = 122942.42196571268

        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field3LUTColorBar = GetScalarBar(field3LUT, renderView1)
        field3LUTColorBar.Orientation = 'Horizontal'
        field3LUTColorBar.WindowLocation = 'Any Location'
        field3LUTColorBar.Position = [0.041, 0.908]
        field3LUTColorBar.ScalarBarLength = 0.33
        field3LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field3LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field3LUTColorBar.TitleFontFamily = 'Times'
        field3LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field3LUT = GetColorTransferFunction(field3)
        field3PWF = GetOpacityTransferFunction(field3)
        field3LUT.RescaleTransferFunction(0.0, 1.0)
        field3PWF.RescaleTransferFunction(0.0, 1.0)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "%s_wedge_t%.4e.pdf" % (field3, self.time))
        fig_png_path = os.path.join(self.pv_output_dir, "%s_wedge_t%.4e.png" % (field3, self.time))
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # hide plots
        # Hide(source1, renderView1)
        # Hide(sourceV, renderView1)
        # Hide(sourceVRE, renderView1)
        # Hide(sourceVTXT, renderView1)
        # Hide(source_contour1, renderView1)
        # HideScalarBarIfNotNeeded(field2LUT, renderView1)
        # HideScalarBarIfNotNeeded(field3LUT, renderView1)
        # HideScalarBarIfNotNeeded(fieldVLUT, renderView1)
        # HideScalarBarIfNotNeeded(fieldTLUT, renderView1)
    
    def plot_step_wedge_02252025(self, **kwargs): 
        '''
        plot a step and focus on the wedge
        Inputs:
            kwargs:
                glyphRegistrationName: the name of the glyph, used to adjust the glyph outputs
        '''
        # parse input
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
        
        # get active view and source
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]

        # set source and field
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"
        field2 = "viscosity"
        field3 = "spcrust"
        layout_resolution = (1350, 704)
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)

        # glyph scales
        # For the wedge plot and the "small" wedge plot
        # we use different method. 
        scale_factor = 2.5e5
        n_sample_points = 400000
        n_stride_small = 150
        CameraParallelScale = 101605.30740967988
        CameraParallelScaleSmall = 39173.24444222669
        small_scale_fraction = CameraParallelScaleSmall / CameraParallelScale
       
        # find source
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)
    
        # show source1
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
        if PLOT_AXIS:
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        
        # Configure temperature field settings for the scalar plot.
        field1LUT = GetColorTransferFunction(field1)
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)

        # colorbar position
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        
        # show sourceV (vector field)
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
        # get color transfer function/color map for 'field'
        fieldVLUT = GetColorTransferFunction('velocity')
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)
        # colorbar position
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0


        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]

        # Set Camera parameters
        camera_x = -192.1742524223165

        # Set Glyph
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [camera_x - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        

        # Show contour
        fieldTLUT = GetColorTransferFunction("T")
        source_contour = FindSource("ContourT1")

        contourTDisplay = Show(source_contour, renderView1, 'GeometryRepresentation')
        
        rescale_transfer_function_combined('T', 273.15, 1273.15) # change this range to adjust temperature
        
        # Show contour1: spcrust 
        source_contour1 = FindSource("ContourCr")
        contourCrDisplay = Show(source_contour1, renderView1, 'GeometryRepresentation')

        
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6300734.12934143, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6300734.12934143, 0.0]
            renderView1.CameraParallelScale = CameraParallelScale
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "T_wedge_02252025_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_wedge_02252025_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # adjust for a smaller region
        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
        # set the camera position
        CameraPosition = [-5144.780289291184, 6305365.90238875, 25000000.0]
        CameraFocalPoint = [-5144.780289291184, 6305365.90238875, 0.0]
        # Set Glyph
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [CameraPosition[0] - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor * small_scale_fraction, n_stride_small,\
            point_source_center, GlyphMode="Every Nth Point")
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScaleSmall
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "T_wedge_small_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_wedge_small_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # Comment the codes below to debug
        # second plot
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # adjust glyph 
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [camera_x - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6300734.12934143, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6300734.12934143, 0.0]
            renderView1.CameraParallelScale = CameraParallelScale
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field2LUTColorBar = GetScalarBar(field2LUT, renderView1)
        field2LUTColorBar.Orientation = 'Horizontal'
        field2LUTColorBar.WindowLocation = 'Any Location'
        field2LUTColorBar.Position = [0.041, 0.908]
        field2LUTColorBar.ScalarBarLength = 0.33
        field2LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field2LUTColorBar.TitleFontFamily = 'Times'
        field2LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field2LUT = GetColorTransferFunction(field2)
        field2PWF = GetOpacityTransferFunction('viscosity')
        field2LUT.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        field2PWF.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_wedge_02252025_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_wedge_02252025_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # adjust for a smaller region
        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
        # set the camera position
        CameraPosition = [-5144.780289291184, 6305365.90238875, 25000000.0]
        CameraFocalPoint = [-5144.780289291184, 6305365.90238875, 0.0]
        # Set Glyph
        # n_sample_points = 1600000
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [CameraPosition[0] - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor * small_scale_fraction, n_stride_small,\
            point_source_center, GlyphMode="Every Nth Point")

        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScaleSmall
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_wedge_small_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_wedge_small_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        
        # third plot
        # set scalar coloring
        ColorBy(source1Display, ('POINTS', field3, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        # Adjust glyph 
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [camera_x - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6300734.12934143, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6300734.12934143, 0.0]
            renderView1.CameraParallelScale = CameraParallelScale

        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field3LUTColorBar = GetScalarBar(field3LUT, renderView1)
        field3LUTColorBar.Orientation = 'Horizontal'
        field3LUTColorBar.WindowLocation = 'Any Location'
        field3LUTColorBar.Position = [0.041, 0.908]
        field3LUTColorBar.ScalarBarLength = 0.33
        field3LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field3LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field3LUTColorBar.TitleFontFamily = 'Times'
        field3LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        field3LUT = GetColorTransferFunction(field3)
        field3PWF = GetOpacityTransferFunction(field3)
        field3LUT.RescaleTransferFunction(0.0, 1.0)
        field3PWF.RescaleTransferFunction(0.0, 1.0)
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "spcrust_wedge_02252025_t%.4e.pdf" % (self.time))
        fig_png_path = os.path.join(self.pv_output_dir, "spcrust_wedge_02252025_t%.4e.png" % (self.time))
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # adjust for a smaller region
        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
        # set the camera position
        CameraPosition = [-5144.780289291184, 6305365.90238875, 25000000.0]
        CameraFocalPoint = [-5144.780289291184, 6305365.90238875, 0.0]
        # Set Glyph
        # n_sample_points = 1600000
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [CameraPosition[0] - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor * small_scale_fraction, n_stride_small,\
            point_source_center, GlyphMode="Every Nth Point")
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScaleSmall
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "spcrust_wedge_small_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "spcrust_wedge_small_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)


        # fourth plot
        # set scalar coloring
        field4 = "strain_rate"
        ColorBy(source1Display, ('POINTS', field4, 'Magnitude'))
        source1Display.SetScalarBarVisibility(renderView1, True)
        # hide the grid axis
        renderView1.OrientationAxesVisibility = 0
        # Hide the scalar bar for the first field color map
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        # Adjust glyph 
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [camera_x - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(1350, 704)
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = [camera_x, 6300734.12934143, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6300734.12934143, 0.0]
            renderView1.CameraParallelScale = CameraParallelScale
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # colorbar position
        field4LUT = GetColorTransferFunction(field4)
        field4LUTColorBar = GetScalarBar(field4LUT, renderView1)
        field4LUTColorBar.Orientation = 'Horizontal'
        field4LUTColorBar.WindowLocation = 'Any Location'
        field4LUTColorBar.Position = [0.041, 0.908]
        field4LUTColorBar.ScalarBarLength = 0.33
        field4LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field4LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field4LUTColorBar.TitleFontFamily = 'Times'
        field4LUTColorBar.LabelFontFamily = 'Times'
        # get color transfer function/color map for 'field'
        # convert to log space
        strain_rate_min = 1e-16
        strain_rate_max = 1e-14
        field4LUT.ApplyPreset("Viridis (matplotlib)", True)
        field4LUT = GetColorTransferFunction(field4)
        field4PWF = GetOpacityTransferFunction(field4)
        field4LUT.RescaleTransferFunction(strain_rate_min, strain_rate_max)
        field4PWF.RescaleTransferFunction(strain_rate_min, strain_rate_max)
        field4LUT.MapControlPointsToLogSpace()
        field4LUT.UseLogScale = 1
        field4PWF.MapControlPointsToLogSpace()
        field4PWF.UseLogScale = 1
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "strain_rate_wedge_02252025_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "strain_rate_wedge_02252025_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # adjust for a smaller region
        # change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
        # show the representative point                                                                                                    
        _source_v_re = _source_v + "_representative"                                                                                       
        sourceVRE = FindSource(_source_v_re)                                                                                               
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation') 
        # show the annotation
        _source_v_txt = _source_v + "_text" 
        sourceVTXT = FindSource(_source_v_txt)                                                                                               
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
        # set the camera position
        CameraPosition = [-5144.780289291184, 6305365.90238875, 25000000.0]
        CameraFocalPoint = [-5144.780289291184, 6305365.90238875, 0.0]
        # Set Glyph
        point_source_center = [0.0, 0.0, 0.0]
        if "chunk" == "chunk":
            point_source_center = [CameraPosition[0] - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
        self.adjust_glyph_properties('Glyph1', scale_factor * small_scale_fraction, n_stride_small,\
            point_source_center, GlyphMode="Every Nth Point")
        # adjust layout and camera & get layout & set layout/tab size in pixels
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
        if "GEOMETRY" == "chunk":
            # test new camera parameters
            renderView1.CameraPosition = CameraPosition
            renderView1.CameraFocalPoint = CameraFocalPoint
            renderView1.CameraParallelScale = CameraParallelScaleSmall
        elif "GEOMETRY" == "box":
            raise NotImplementError()
        # save figure
        fig_path = os.path.join(self.pv_output_dir, "strain_rate_wedge_small_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "strain_rate_wedge_small_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)


        # hide plots
        # Hide(source1, renderView1)
        # Hide(sourceV, renderView1)
        # Hide(sourceVRE, renderView1)
        # Hide(sourceVTXT, renderView1)
        # Hide(source_contour, renderView1)
        # Hide(source_contour1, renderView1)
        # HideScalarBarIfNotNeeded(field2LUT, renderView1)
        # HideScalarBarIfNotNeeded(field3LUT, renderView1)
        # HideScalarBarIfNotNeeded(fieldVLUT, renderView1)
        # HideScalarBarIfNotNeeded(fieldTLUT, renderView1)

    def plot_step_wedge1(self, **kwargs):
        '''
        Plot a step and focus on the wedge
        Inputs:
            kwargs:
                glyphRegistrationName: the name of the glyph, used to adjust the glyph outputs
        '''
        # Parse input
        glyphRegistrationName = kwargs.get("glyphRegistrationName", "Glyph1")
    
        # Get active view and source
        renderView1 = GetActiveViewOrCreate('RenderView')
        renderView1.UseColorPaletteForBackground = 0
        renderView1.Background = [1.0, 1.0, 1.0]
    
        # Set source and field
        _source = "Transform1"
        _source_v = "Glyph1"
        field1 = "T"
        field2 = "viscosity"
        field3 = "spcrust"
        layout_resolution = (1350, 704)
    
        # Get color transfer function/color map for fields
        field2LUT = GetColorTransferFunction(field2)
        field3LUT = GetColorTransferFunction(field3)
    
        # Find source
        source1 = FindSource(_source)
        sourceV = FindSource(_source_v)
    
        # Show source1
        source1Display = Show(source1, renderView1, 'GeometryRepresentation')
        source1Display.SetScalarBarVisibility(renderView1, True)
    
        if PLOT_AXIS:
            source1Display.DataAxesGrid.GridAxesVisibility = 1
            source1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            source1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

        # Plot field1 
        # Get color transfer function for field1
        field1LUT = GetColorTransferFunction(field1)
    
        # Set scalar coloring
        ColorBy(source1Display, ('POINTS', field1, 'Magnitude'))
        HideScalarBarIfNotNeeded(field2LUT, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)
    
        # Rescale transfer function
        field1LUT.RescaleTransferFunction(273.0, 1673.0)
    
        # Colorbar position
        field1LUTColorBar = GetScalarBar(field1LUT, renderView1)
        field1LUTColorBar.Orientation = 'Horizontal'
        field1LUTColorBar.WindowLocation = 'Any Location'
        field1LUTColorBar.Position = [0.041, 0.908]
        field1LUTColorBar.ScalarBarLength = 0.33
        field1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        field1LUTColorBar.TitleFontFamily = 'Times'
        field1LUTColorBar.LabelFontFamily = 'Times'
    
        # Hide the grid axis
        renderView1.OrientationAxesVisibility = 0
    
        # Show sourceV (vector field)
        sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
        sourceVDisplay.SetScalarBarVisibility(renderView1, True)
    
        # Get color transfer function for velocity
        fieldVLUT = GetColorTransferFunction('velocity')
    
        if MAX_VELOCITY > 0.0:
            fieldVLUT.RescaleTransferFunction(0.0, MAX_VELOCITY)
    
        # Colorbar position for velocity
        fieldVLUTColorBar = GetScalarBar(fieldVLUT, renderView1)
        fieldVLUTColorBar.Orientation = 'Horizontal'
        fieldVLUTColorBar.WindowLocation = 'Any Location'
        fieldVLUTColorBar.Position = [0.630, 0.908]
        fieldVLUTColorBar.ScalarBarLength = 0.33
        fieldVLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
        fieldVLUTColorBar.TitleFontFamily = 'Times'
        fieldVLUTColorBar.LabelFontFamily = 'Times'
    
        # Hide the grid axis
        renderView1.OrientationAxesVisibility = 0
    
        # Change point position
        pointName = "PointSource_" + glyphRegistrationName
        pointSource1 = FindSource(pointName)
    
        if "chunk" == "chunk":
            pointSource1.Center = [0, 6.7e6, 0]
    
        # Show the representative point
        _source_v_re = _source_v + "_representative"
        sourceVRE = FindSource(_source_v_re)
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation')
    
        # Show the annotation
        _source_v_txt = _source_v + "_text"
        sourceVTXT = FindSource(_source_v_txt)
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]
    
        # Adjust glyph properties
        scale_factor = 5e5
        n_sample_points = 100000
        camera_x = 0.0
        point_source_center = [0.0, 0.0, 0.0]

        # Show contour
        fieldTLUT = GetColorTransferFunction("T")
        source_contour = FindSource("ContourT1")

        contourTDisplay = Show(source_contour, renderView1, 'GeometryRepresentation')
        
        rescale_transfer_function_combined('T', 273.0, 1673.0)

        # Show contour1: spcrust 
        source_contour1 = FindSource("ContourCr")
        contourCrDisplay = Show(source_contour1, renderView1, 'GeometryRepresentation')
    
        if "chunk" == "chunk":
            point_source_center = [camera_x - 0.1e5, 6.4e6, 0]
        elif "chunk" == "box":
            point_source_center = [4.65e6, 2.95e6, 0]
        else:
            raise NotImplementedError()
    
        self.adjust_glyph_properties('Glyph1', scale_factor, n_sample_points, point_source_center)
    
        # Adjust layout and camera
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
    
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [camera_x, 6259332.1857954515, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6259332.1857954515, 0.0]
            renderView1.CameraParallelScale = 175692.00000000006
        elif "GEOMETRY" == "box":
            raise NotImplementedError()
    
        # Save figure
        fig_path = os.path.join(self.pv_output_dir, "T_wedge_bigger_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "T_wedge_bigger_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)

        # todo_field2 
        # Plot field2
        # Get color transfer function for field2
        field2LUT = GetColorTransferFunction(field2)
    
        # Set scalar coloring
        ColorBy(source1Display, ('POINTS', field2, 'Magnitude'))
        HideScalarBarIfNotNeeded(field1LUT, renderView1)
        source1Display.SetScalarBarVisibility(renderView1, True)

        # rescale color functions 
        field2PWF = GetOpacityTransferFunction('viscosity')
        field2LUT.RescaleTransferFunction(ETA_MIN, ETA_MAX)
        field2PWF.RescaleTransferFunction(ETA_MIN, ETA_MAX)

        # Adjust layout and camera
        layout1 = GetLayout()
        layout1.SetSize(layout_resolution[0], layout_resolution[1])
        renderView1.InteractionMode = '2D'
    
        if "GEOMETRY" == "chunk":
            renderView1.CameraPosition = [camera_x, 6259332.1857954515, 25000000.0]
            renderView1.CameraFocalPoint = [camera_x, 6259332.1857954515, 0.0]
            renderView1.CameraParallelScale = 175692.00000000006
        elif "GEOMETRY" == "box":
            raise NotImplementedError()
    
        # Save figure
        fig_path = os.path.join(self.pv_output_dir, "viscosity_wedge_bigger_t%.4e.pdf" % self.time)
        fig_png_path = os.path.join(self.pv_output_dir, "viscosity_wedge_bigger_t%.4e.png" % self.time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)
    
    
        # Hide plots
        Hide(source1, renderView1)
        Hide(sourceV, renderView1)
        Hide(sourceVRE, renderView1)
        Hide(sourceVTXT, renderView1)
        Hide(source_contour, renderView1)
        Hide(source_contour1, renderView1)
        HideScalarBarIfNotNeeded(field3LUT, renderView1)
        HideScalarBarIfNotNeeded(fieldVLUT, renderView1)


    def adjust_glyph_properties(self, registrationName, scale_factor, n_value, point_center, **kwargs):
        '''
        adjust the properties of the glyph source
        Inputs:
            scale_factor: scale of arrows
            n_value: used to assign the MaximumNumberOfSamplePoints or Stride variable.
            ghost_field: the colorbar of a previous "ghost field" needs to be hide again to
                prevent it from being shown.
            kwargs:
                registrationName : the name of registration
                representative_value: a value to represent by the constant vector
                GlyphMode: mode of glyph
        '''
        GlyphMode = kwargs.get("GlyphMode", "Uniform Spatial Distribution (Bounds Based)")

        assert(type(n_value) == int)
        assert(type(point_center) == list and len(point_center) == 3)
        representative_value = kwargs.get("representative_value", 0.05)
        
        glyph1 = FindSource(registrationName)
        glyph1.ScaleFactor = scale_factor

        if GlyphMode == "Uniform Spatial Distribution (Bounds Based)":
            glyph1.GlyphMode = GlyphMode
            glyph1.MaximumNumberOfSamplePoints = n_value
        elif GlyphMode == "Every Nth Point":
            glyph1.GlyphMode = GlyphMode
            glyph1.Stride = n_value
        else:
            return NotImplementedError()

    
        pointName = "PointSource_" + registrationName
        pointSource1 = FindSource(pointName)
        pointSource1.Center = point_center
    
        calculatorName="Calculator_" + registrationName
        calculator1 = FindSource(calculatorName)
        calculator1.Function = '%.4e*iHat' % (scale_factor*representative_value)
    
        textName = registrationName + "_text"
        text1 = FindSource(textName)


def add_glyph1(_source, field, scale_factor, **kwargs):
    '''
    add glyph in plots
    Inputs:
        scale_factor: scale of arrows
        ghost_field: the colorbar of a previous "ghost field" needs to be hide again to
            prevent it from being shown.
        kwargs:
            registrationName : the name of registration
            representative_value: a value to represent by the constant vector
    '''
    registrationName = kwargs.get("registrationName", 'Glyph1')
    representative_value = kwargs.get("representative_value", 0.05)
    
    # get active source and renderview
    pvd = FindSource(_source)
    renderView1 = GetActiveViewOrCreate('RenderView')

    # add glyph
    glyph1 = Glyph(registrationName=registrationName, Input=pvd, GlyphType='2D Glyph')
    # adjust orientation and scale
    glyph1.OrientationArray = ['POINTS', 'velocity']
    glyph1.ScaleArray = ['POINTS', 'velocity']
    glyph1.ScaleFactor = scale_factor
    glyph1.MaximumNumberOfSamplePoints = 20000

    glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
    field0 = glyph1Display.ColorArrayName[1]
    field0LUT = GetColorTransferFunction(field0)
    # set the vector line width
    glyph1Display.LineWidth = 2.0
    # show color bar/color legend
    glyph1Display.SetScalarBarVisibility(renderView1, True)
    # get color transfer function/color map for 'field'
    fieldLUT = GetColorTransferFunction(field)
    # get opacity transfer function/opacity map for 'field'
    fieldPWF = GetOpacityTransferFunction(field)

    # set scalar coloring
    # ColorBy(glyph1Display, None) # this doesn't work anymore
    glyph1Display.AmbientColor = [1.0, 1.0, 1.0]
    glyph1Display.DiffuseColor = [1.0, 1.0, 1.0]

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)
    fieldLUT1 = GetColorTransferFunction(field0)
    HideScalarBarIfNotNeeded(fieldLUT1, renderView1)
    
    # add a representative vector
    pointName = "PointSource_" + registrationName
    pointSource1 = PointSource(registrationName=pointName)
    if "GEOMETRY" == "chunk":
        pointSource1.Center = [0, 6.4e6, 0]
    elif "GEOMETRY" == "box":
        pointSource1.Center = [4.65e6, 2.95e6, 0]
        
    # pointSource1.Center 
    pointSource1Display = Show(pointSource1, renderView1, 'GeometryRepresentation')
    print(dir(pointSource1))
    print(pointSource1.Center)

    calculatorName="Calculator_" + registrationName
    calculator1 = Calculator(registrationName=calculatorName, Input=pointSource1)
    calculator1.ResultArrayName = 'constant_velocity'
    calculator1.Function = '%.4e*iHat' % (scale_factor*representative_value)
    calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')

    # add glyph
    glyph2Name = registrationName+"_representative"
    glyph2 = Glyph(registrationName=glyph2Name, Input=calculator1, GlyphType='2D Glyph')
    # adjust orientation and scale
    glyph2.OrientationArray = ['POINTS', 'constant_velocity']
    # glyph2.ScaleArray = ['POINTS', 'No scale array']
    glyph2.ScaleArray = ['POINTS', 'constant_velocity']
    # glyph2.ScaleFactor = 4e4
    glyph2.ScaleFactor = 1.0
    glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')
    glyph2Display.AmbientColor = [0.0, 0.0, 0.0]
    glyph2Display.DiffuseColor = [0.0, 0.0, 0.0]
    # set the vector line width
    glyph2Display.LineWidth = 2.0
    # show color bar/color legend
    glyph2Display.SetScalarBarVisibility(renderView1, False)

    # add text
    # create a new 'Text'
    textName = registrationName + "_text"
    text1 = Text(registrationName=textName)
    # Properties modified on text1
    text1.Text = '5cm / yr'
    # show data in view
    text1Display = Show(text1, renderView1, 'TextSourceRepresentation')
    # Properties modified on text1Display
    text1Display.WindowLocation = 'Upper Center'
    text1Display.Color = [0.0, 0.0, 0.0]
    
    # hide data in view
    # Hide(pvd, renderView1)
    # hide glaph in view
    Hide(glyph1, renderView1)
    Hide(pointSource1, renderView1)
    Hide(calculator1)
    Hide(glyph2, renderView1)
    Hide(text1, renderView1)
    
    # update the view to ensure updated data information
    renderView1.Update()

# todo_dd
def add_deformation_mechanism(_source, **kwargs):
    '''
    add programable filter to deal with the deformation mechanism
    Inputs:
        _source to extract the data
    kwargs:
        registrationName : the name of registration
    '''
    registrationName = kwargs.get("registrationName", 'Glyph1')
    
    # get active source and renderview
    pvd = FindSource(_source)
    renderView1 = GetActiveViewOrCreate('RenderView')

    # create a new 'Programmable Filter'
    programmableFilter1 = ProgrammableFilter(registrationName=registrationName, Input=pvd)
    programmableFilter1.Script = \
"""
import numpy as np
pressure = inputs[0].PointData["p"].GetArrays()[0]
strain_rate = inputs[0].PointData["strain_rate"].GetArrays()[0]
diff_visc = inputs[0].PointData["diffusion_viscosity"].GetArrays()[0]
disl_visc= inputs[0].PointData["dislocation_viscosity"].GetArrays()[0]
p_visc= inputs[0].PointData["peierls_viscosity"].GetArrays()[0]
y_visc1 = (pressure * np.sin(25.0/180.0 * np.pi) + 50e6) / 2.0 / strain_rate
y_visc2 = 500e6 / 2.0 / strain_rate
y_visc = np.minimum(y_visc1, y_visc2)
d_mech = np.argmin(np.array([diff_visc, disl_visc, y_visc, p_visc]), axis=0)
output.PointData.append(d_mech, "deformation_mechanism")
"""
    programmableFilter1.RequestInformationScript = ''
    programmableFilter1.RequestUpdateExtentScript = ''
    programmableFilter1.PythonPath = ''

# todo_Vres
def add_velocity_residual(_source, **kwargs):
    '''
    add programable filter to deal with the deformation mechanism
    Inputs:
        _source to extract the data
    kwargs:
        registrationName : the name of registration
    '''
    registrationName = kwargs.get("registrationName", 'Vres')
    
    # get active source and renderview
    pvd = FindSource(_source)
    renderView1 = GetActiveViewOrCreate('RenderView')

    # create a new 'Programmable Filter'
    programmableFilter = ProgrammableFilter(registrationName=registrationName, Input=pvd)
    programmableFilter.Script = \
"""
from paraview.vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np

vtk_points = inputs[0].GetPoints()
np_points = vtk_to_numpy(vtk_points)

# Split into X, Y, Z
x = np_points[:, 0]
y = np_points[:, 1]

r = np.sqrt(x**2 + y**2)
phi = np.mod(np.arctan2(y, x), 2*np.pi)           # azimuthal angle [-pi, pi]

v1phi = 10
v1x = -v1phi * np.sin(phi)
v1y = v1phi * np.cos(phi)

# 6. Load original velocity field
velocity = vtk_to_numpy(inputs[0].GetPointData().GetArray("velocity"))
vx, vy, vz = velocity[:, 0], velocity[:, 1], velocity[:, 2]

# 7. Subtract φ-direction velocity
vx_res = vx - v1x
vy_res = vy - v1y
vz_res = vz

# 8. Combine into residual velocity array
velocity_residual = np.column_stack((vx_res, vy_res, vz_res))

# 9. Convert to VTK array and add to output
vtk_array = numpy_to_vtk(velocity_residual, deep=True)
vtk_array.SetName("velocity_residual")
output.GetPointData().AddArray(vtk_array)
"""
    programmableFilter.RequestInformationScript = ''
    programmableFilter.RequestUpdateExtentScript = ''
    programmableFilter.PythonPath = ''



def main():
    all_available_graphical_snapshots = ALL_AVAILABLE_GRAPHICAL_SNAPSHOTS
    all_available_graphical_times = ALL_AVAILABLE_GRAPHICAL_TIMES
    # types of plots included
    plot_types = PLOT_TYPES
    assert(len(all_available_graphical_snapshots) == len(all_available_graphical_times))
    # First, make directory for images if it's not there
    if not os.path.isdir("IMG_OUTPUT_DIR"):
        os.mkdir("IMG_OUTPUT_DIR")
    # Process and generate plots
    Slab = SLAB("PARAVIEW_FILE", output_dir="IMG_OUTPUT_DIR")
    # First number is the number of initial adaptive refinements
    # Second one is the snapshot to plot
    # here we prefer to use a series of snapshots.
    # If this doesn't work, we will use a single snapshot
    steps = GRAPHICAL_STEPS
    if not steps == []:
        for step in steps:
            # check that snapshot is valid
            snapshot = INITIAL_ADAPTIVE_REFINEMENT+step
            if snapshot in all_available_graphical_snapshots:
                idx = all_available_graphical_snapshots.index(snapshot)
                _time =  all_available_graphical_times[idx]
                Slab.goto_time(_time)
                if "upper_mantle" in plot_types:
                    Slab.plot_step_upper_mantle()
                if "upper_mantle_DM" in plot_types:
                    Slab.plot_step_upper_mantle_DM()
                if "whole" in plot_types:
                    Slab.plot_step_whole()
                if "whole_whole" in plot_types:
                    Slab.plot_step_whole_whole()
                if "wedge" in plot_types:
                    Slab.plot_step_wedge()
                if "wedge_02252025" in plot_types:
                    Slab.plot_step_wedge_02252025()
                if "wedge_bigger" in plot_types:
                    Slab.plot_step_wedge1()
                if "wedge_small" in plot_types:
                    Slab.plot_step_wedge_small()
               #  Slab(INITIAL_ADAPTIVE_REFINEMENT+step)
            else:
                print ("step %s is not valid. There is no output" % step)
    else:
        snapshot = SINGLE_SNAPSHOT
        idx = all_available_graphical_snapshots.index(snapshot)
        _time =  all_available_graphical_times[idx]
        Slab.goto_time(_time)
        # Slab.plot_slice()
        # Slab(INITIAL_ADAPTIVE_REFINEMENT+SINGLE_SNAPSHOT)


def ggr2cart(lat,lon,r):
    # transform spherical lat,lon,r geographical coordinates
    # to global cartesian xyz coordinates
    #
    # input:  lat,lon,r in radians, meters
    # output: x,y,z in meters 3 x M
    sla = math.sin(lat)
    cla = math.cos(lat)
    slo = math.sin(lon)
    clo = math.cos(lon)

    x = r * cla * clo
    y = r * cla * slo
    z = r * sla

    return x,y,z


main()
