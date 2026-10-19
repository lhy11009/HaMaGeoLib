def set_viscosity_plot(sourceDisplay, eta_min, eta_max):
    '''
    set the viscosity plot
    Inputs:
        eta_min (float): minimum viscosity
        eta_max (float): maximum viscosity
    '''
    field = "viscosity"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, eta_min, eta_max)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.MapControlPointsToLogSpace()
    fieldLUT.UseLogScale = 1
    fieldLUT.ApplyPreset("Viridis", True)
    fieldLUT.InvertTransferFunction()
    return field, fieldLUT


def set_viscosity_colorbar(renderView, fieldLUT, *, 
                           position=None):
    '''
    set the viscosity colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for viscosity
        position - manual position of colorbar
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Viscosity'
    scalarBar.ComponentTitle = '[Pa·s]'
    scalarBar.Orientation = 'Horizontal'

    if position is not None:
        scalarBar.WindowLocation = 'Any Location'
        scalarBar.Position = position   # (x, y) in normalized viewport coordinates
    return scalarBar


def set_temperature_plot(sourceDisplay):
    '''
    set the temperature plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    field = "T"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, 273.15, 2273.15)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("SCM_lapaz", True)
    return field, fieldLUT


def set_temperature_colorbar(renderView, fieldLUT):
    '''
    set the temperature colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for temperature
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Temperature'
    scalarBar.ComponentTitle = '[K]'
    scalarBar.Orientation = 'Horizontal'
    return scalarBar


def set_density_plot(sourceDisplay):
    '''
    set the viscosity plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    field = "density"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, 3000.0, 4000.0)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("SCM_bilbao", True)
    return field, fieldLUT


def set_density_colorbar(renderView, fieldLUT):
    '''
    set the density colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for density
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Density'
    scalarBar.ComponentTitle = '[kg/m^3]'
    scalarBar.Orientation = 'Horizontal'
    return scalarBar


def set_pin_line_plot(sourceDisplay):
    '''
    set the pin line plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    field = "initial_depth"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, 0.0, 40e3)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("SCM_grayC", True)
    fieldLUT.InvertTransferFunction()
    return field, fieldLUT


def set_active_deforming_plot(sourceDisplay):
    '''
    set the active deforming plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    ColorBy(sourceDisplay, None)
    sourceDisplay.DiffuseColor = [36/255.0, 0/255.0, 107/255.0] # 24006b
    sourceDisplay.Opacity = 0.5

TPOGRAPHY_COLOR_POINTS = [
    (0.0, 35 / 255.0, 27 / 255.0, 73 / 255.0),
    ((27 - 9) / (197 - 9), 45 / 255.0, 51 / 255.0, 125 / 255.0),
    ((46 - 9) / (197 - 9), 30 / 255.0, 77 / 255.0, 158 / 255.0),
    ((65 - 9) / (197 - 9), 38 / 255.0, 102 / 255.0, 174 / 255.0),
    ((84 - 9) / (197 - 9), 46 / 255.0, 128 / 255.0, 197 / 255.0),
    ((103 - 9) / (197 - 9), 50 / 255.0, 155 / 255.0, 217 / 255.0),
    ((122 - 9) / (197 - 9), 72 / 255.0, 180 / 255.0, 230 / 255.0),
    ((141 - 9) / (197 - 9), 81 / 255.0, 202 / 255.0, 254 / 255.0),
    ((150 - 9) / (197 - 9), 151 / 255.0, 201 / 255.0, 159 / 255.0),
    ((157 - 9) / (197 - 9), 206 / 255.0, 175 / 255.0, 96 / 255.0),
    ((165 - 9) / (197 - 9), 166 / 255.0, 91 / 255.0, 55 / 255.0),
    ((174 - 9) / (197 - 9), 130 / 255.0, 86 / 255.0, 90 / 255.0),
    ((183 - 9) / (197 - 9), 120 / 255.0, 108 / 255.0, 109 / 255.0),
    (1.0, 251 / 255.0, 251 / 255.0, 251 / 255.0),
]

# Sedimentation-positive rate: erosion is blue, sedimentation is red.
EROSION_DEPOSITION_COLOR_POINTS = [
    (0.00, 37 / 255.0, 52 / 255.0, 148 / 255.0),
    (0.25, 44 / 255.0, 127 / 255.0, 184 / 255.0),
    (0.40, 127 / 255.0, 205 / 255.0, 187 / 255.0),
    (0.50, 247 / 255.0, 247 / 255.0, 247 / 255.0),
    (0.60, 253 / 255.0, 174 / 255.0, 97 / 255.0),
    (0.75, 215 / 255.0, 48 / 255.0, 39 / 255.0),
    (1.00, 127 / 255.0, 0 / 255.0, 0 / 255.0),
]

# Cumulative fields retain their deposition-negative sign convention.
CUMULATIVE_EROSION_DEPOSITION_COLOR_POINTS = [
    (0.00, 127 / 255.0, 0 / 255.0, 0 / 255.0),
    (0.25, 215 / 255.0, 48 / 255.0, 39 / 255.0),
    (0.40, 253 / 255.0, 174 / 255.0, 97 / 255.0),
    (0.50, 247 / 255.0, 247 / 255.0, 247 / 255.0),
    (0.60, 127 / 255.0, 205 / 255.0, 187 / 255.0),
    (0.75, 44 / 255.0, 127 / 255.0, 184 / 255.0),
    (1.00, 37 / 255.0, 52 / 255.0, 148 / 255.0),
]


def add_optional_fastscape_contour(source, field, value, render_view, **kwargs):
    """Add a FastScape contour when contour plotting is enabled."""
    if not PLOT_FASTSCAPE_CONTOURS:
        return None
    return add_dashed_contour(source, field, value, render_view, **kwargs)


def remove_optional_fastscape_contour(contour_filters, render_view):
    """Remove optional FastScape contour filters when they were created."""
    if contour_filters is not None:
        remove_dashed_contour(*contour_filters, render_view)


def apply_custom_color_points(fieldLUT, minimum, maximum, color_points):
    """Apply normalized RGB control points over a field's numeric range."""
    fieldLUT.UseLogScale = 0
    fieldLUT.ColorSpace = "RGB"
    fieldLUT.RGBPoints = [
        component
        for position, red, green, blue in color_points
        for component in (
            minimum + position * (maximum - minimum),
            red,
            green,
            blue,
        )
    ]


def set_fastscape_topography_plot(sourceDisplay):
    '''
    set the topography colorbar for fastscape outputs
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for viscosity
    ''' 
    field = "topography"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, -11000.0, 4000.0)
    fieldLUT = GetColorTransferFunction(field)
    apply_custom_color_points(
        fieldLUT, -11000.0, 4000.0, TPOGRAPHY_COLOR_POINTS
    )
    return field, fieldLUT

def set_fastscape_topography_colorbar(renderView, fieldLUT, *,
                                      position=None):
    '''
    set the fastscape topography colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for density
        position - manual position of colorbar
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Topography'
    scalarBar.ComponentTitle = '[m]'
    scalarBar.Orientation = 'Horizontal'

    if position is not None:
        scalarBar.WindowLocation = 'Any Location'
        scalarBar.Position = position   # (x, y) in normalized viewport coordinates
    return scalarBar

def set_fastscape_erosion_rate_plot(sourceDisplay):
    '''
    set the erosion rate plot for fastscape outputs
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for viscosity
    ''' 
    field = "signed_sedimentation_rate"
    if USE_LEGACY_EROSION_RATE_RANGE:
        minimum, maximum = -1e-2, 1e-2
    else:
        minimum, maximum = -3e-4, 3e-4
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, minimum, maximum)
    fieldLUT = GetColorTransferFunction(field)
    apply_custom_color_points(
        fieldLUT, minimum, maximum, EROSION_DEPOSITION_COLOR_POINTS
    )
    return field, fieldLUT

def set_fastscape_erosion_rate_colorbar(renderView, fieldLUT, *,
                                      position=None):
    '''
set the fastscape erosion rate colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for density
        position - manual position of colorbar
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Erosion (-) / sedimentation (+) rate'
    scalarBar.ComponentTitle = '[m/yr]'
    scalarBar.Orientation = 'Horizontal'

    if position is not None:
        scalarBar.WindowLocation = 'Any Location'
        scalarBar.Position = position   # (x, y) in normalized viewport coordinates
    return scalarBar

def set_fastscape_total_erosion_plot(sourceDisplay):
    '''
    set the total erosion plot for fastscape outputs
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for viscosity
    ''' 
    field = "total_erosion"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, -5e3, 5e3)
    fieldLUT = GetColorTransferFunction(field)
    apply_custom_color_points(
        fieldLUT, -5e3, 5e3, CUMULATIVE_EROSION_DEPOSITION_COLOR_POINTS
    )
    return field, fieldLUT

def set_fastscape_total_erosion_colorbar(renderView, fieldLUT, *,
                                      position=None):
    '''
    set the fastscape total erosion colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for density
        position - manual position of colorbar
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Cumulative deposition (-) / erosion (+)'
    scalarBar.ComponentTitle = '[m]'
    scalarBar.Orientation = 'Horizontal'

    if position is not None:
        scalarBar.WindowLocation = 'Any Location'
        scalarBar.Position = position   # (x, y) in normalized viewport coordinates
    return scalarBar


def set_fastscape_total_sedimentation_plot(sourceDisplay):
    '''Set the cumulative marine sedimentation plot for FastScape outputs.'''
    field = "sedimentation"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, -5e3, 5e3)
    fieldLUT = GetColorTransferFunction(field)
    apply_custom_color_points(
        fieldLUT, -5e3, 5e3, CUMULATIVE_EROSION_DEPOSITION_COLOR_POINTS
    )
    return field, fieldLUT


def set_fastscape_total_sedimentation_colorbar(renderView, fieldLUT, *,
                                                position=None):
    '''Set the cumulative marine sedimentation colorbar.'''
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Cumulative sedimentation (-)'
    scalarBar.ComponentTitle = '[m]'
    scalarBar.Orientation = 'Horizontal'

    if position is not None:
        scalarBar.WindowLocation = 'Any Location'
        scalarBar.Position = position
    return scalarBar



def set_data_axes_grid(renderView1):
    '''
    toggle the data axes for display geologic domain
    Inputs:
        renderView1 - paraview object connect to visualization
    '''
    renderView1.AxesGrid.Visibility = 1

    # if (pv_major > 5) or (pv_major == 5 and pv_minor >= 11):
    #     # ParaView ≥ 5.11: axis scaling is supported
    #     renderView1.AxesGrid.XAxisScale = 1.0e-3
    #     renderView1.AxesGrid.YAxisScale = 1.0e-3
    #     renderView1.AxesGrid.ZAxisScale = 1.0e-3

    #     renderView1.AxesGrid.XTitle = 'X [km]'
    #     renderView1.AxesGrid.YTitle = 'Y [km]'
    #     renderView1.AxesGrid.ZTitle = 'Z [km]'
    # ParaView 5.10 and older: no axis scaling support
    # Keep units honest
    renderView1.AxesGrid.XTitle = 'X [m]'
    renderView1.AxesGrid.YTitle = 'Y [m]'
    renderView1.AxesGrid.ZTitle = 'Z [m]'


def twod_workflow(pv_output_dir, visualization_dir, steps, times, timesteps):
    '''
    Workflow for the twod case
    Inputs:
        pv_output_dir - directory for figure
        visualization_dir - where the original case output locates
        steps - time steps
        times - corresponding times
        timesteps - corresponding timesteps in numerical simulation
    '''
    for i, step in enumerate(steps):
        if PP_INITIAL_REFINEMENT:
            snapshot = INITIAL_ADAPTIVE_REFINEMENT+step
        else:
            snapshot = step
        _time = times[i]
        timestep = timesteps[i]


        # add source
        # The original soure is the "solution" file.
        # The "final" source adds pyvista processed object to the original source and requires running
        # the pyvista workflow first. The idea is to have user specify this in the options by setting "FOO00"
        if FOO00:
            filein = os.path.join("PYVISTA_DIRECTORY", "%05d/final_%05d.vtu" % (snapshot, snapshot)) 
            if not os.path.isfile(filein):
                raise FileExistsError("%s doesn't exist. Needs to run pyvista workflow first.")
            XMLUnstructuredGridReader(registrationName='solution-%05d' % snapshot, FileName=[filein])
        else:
            filein = os.path.join(visualization_dir, "solution-%05d.pvtu" %snapshot) 
            XMLPartitionedUnstructuredGridReader(registrationName='solution-%05d' % snapshot, FileName=[filein])

        # Fastscape source
        # These sources are loaded if input file exists
        # Then, with a Transform filter, it's rotated and translated to the surface of the model.
        # todo_fs
        filein = os.path.join(visualization_dir, "..", "fastscape", "Topography%07d.vtk" % timestep)
        if INCLUDE_FASTSCAPE:
            assert(os.path.isfile(filein))
            LegacyVTKReader(registrationName='fastscape-%05d' % snapshot, FileNames=[filein])
            # todo_pin
            setup_fastscape('fastscape-%05d' % snapshot)

        # Additional sources to load
        # These include a suture profile
        # Profiles of the pinned points
        if FOO01:
            # Add the suture profile
            filein = os.path.join("PYVISTA_DIRECTORY", "%05d/suture_%05d.vtp" % (snapshot, snapshot)) 
            XMLPolyDataReader(registrationName='suture-%05d' % snapshot, FileName=[filein])

            # Add the pin lines and create a threshold based on the "local length"
            filein = os.path.join("PYVISTA_DIRECTORY", "%05d/pin_lines_%05d.vtp" % (snapshot, snapshot)) 
            XMLPolyDataReader(registrationName='pin_lines-%05d' % snapshot, FileName=[filein])
            add_threshold_by_field('pin_lines-%05d' % snapshot, "local_length", [0.0, 20e3], registrationName="pin_lines_threshold1")

            # Add the active deforming region. Ultimately this is from a percentile of strain rate
            if source_has_field('solution-%05d' % snapshot, "active_deforming"):
                add_threshold_by_field('solution-%05d' % snapshot, "active_deforming", [0.8, 1.0], registrationName="solution_active_deforming_threshold1")

        # Main workflow of plotting
        plot_twod_basic("solution-%05d" % snapshot, _time, pv_output_dir)


def plot_twod_basic(source_name, _time, pv_output_dir):
        
    _source = FindSource(source_name)

    lut = GetColorTransferFunction("composition_indicator")

    lut.InterpretValuesAsCategories = 1
    lut.AnnotationsInitialized = 1

    lut.Annotations = [
        "0", "Asthenosphere",
        "1", "Upper crust (sp)",
        "2", "Lower crust (sp)",
        "3", "Oceanic crust",
        "4", "Sea floor sediment",
        "5", "Lithosphere",
        "6", "Upper crust (ov)",
        "7", "Lower crust (ov)",
    ]

    lut.IndexedColors = [
        0.4039, 0.8000, 0.9294,
        0.8118, 0.8157, 0.8235,
        0.9294, 0.4000, 0.4667,
        0.784, 0.353, 0.0,
        1.0, 0.706, 0.314,
        0.1451, 0.5373, 0.2588,
        0.63, 0.605, 0.615,
        0.70, 0.30, 0.35,
    ]

    # programmable_sourceDisplay.ColorArrayName = ["POINTS", "composition_indicator"]
    # programmable_sourceDisplay.LookupTable = lut

    # Hide(programmable_source)

    # add glyph of velocity
    ScaleFactor = None
    if "PLOT_TYPE" == "full_domain":
        ScaleFactor = 2e6
    elif "PLOT_TYPE" == "trench_centered":
        ScaleFactor = 1e6
    elif "PLOT_TYPE" == "orogen":
        ScaleFactor = 1e6
    else:
        raise NotImplementedError()
    add_glyph_with_label(source_name, "velocity", ScaleFactor,
                         registrationName="glyph1",
                         MaximumNumberOfSamplePoints=250,
                         LabelPosition=[RIGHT/2.0, TOP + 200e3, 0.0])
    glyph1 = FindSource("glyph1")
    glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
    label_glyph1 = FindSource("Label_glyph1")
    label_glyph1Display = Show(label_glyph1, renderView1, 'GeometryRepresentation')
    text_glyph1 = FindSource("Text_glyph1")
    text_glyph1Display = Show(text_glyph1, renderView1, 'GeometryRepresentation')
        
    sourceDisplay = Show(_source, renderView1, 'GeometryRepresentation')

    if "PLOT_TYPE" == "full_domain":

        layout_resolution = (1350, 704)

        if ANIMATION:
            # turn on axis grid when making animation
            sourceDisplay.DataAxesGrid.GridAxesVisibility = 1

        layout1 = GetLayout()
        layout1.SetSize((layout_resolution[0], layout_resolution[1]))

        # adjust camera setup
        # Camera position is adjusted relative to the position of the right and top boundary
        renderView1.InteractionMode = '2D'
        renderView1.CameraPosition = [RIGHT/2.0, TOP/2.0, 17717371.391353175]
        renderView1.CameraFocalPoint = [RIGHT/2.0, TOP/2.0, 0.0]
        renderView1.CameraParallelScale = 2588447.7843194483 * RIGHT / 8700e3  # scale to the length
    elif "PLOT_TYPE" == "trench_centered":

        layout_resolution = (1350, 704)

        if ANIMATION:
            # turn on axis grid when making animation
            sourceDisplay.DataAxesGrid.GridAxesVisibility = 1

        layout1 = GetLayout()
        layout1.SetSize((layout_resolution[0], layout_resolution[1]))

        # adjust camera setup
        # Camera position is adjusted relative to the position of the right and top boundary
        # todo_height
        renderView1.InteractionMode = '2D'
        renderView1.CameraPosition = [round(TRENCH_CENTER / 500e3) * 500e3, TOP-279152.305969, 17717371.391353175]
        renderView1.CameraFocalPoint = [round(TRENCH_CENTER / 500e3) * 500e3, TOP-279152.305969, 0.0]
        renderView1.CameraParallelScale = 563321.6543393662  # scale to a 2000 km domain
    elif "PLOT_TYPE" == "orogen":
        # Reorder visibility
        # hide_everything()

        # programmable_source = FindSource("Programmable_comp")
        # sourceDisplay = Show(programmable_source, renderView1, 'GeometryRepresentation')
       
        layout_resolution = (1350, 704)

        if ANIMATION:
            # turn on axis grid when making animation
            sourceDisplay.DataAxesGrid.GridAxesVisibility = 1

        layout1 = GetLayout()
        layout1.SetSize((layout_resolution[0], layout_resolution[1]))

        # adjust camera setup
        # Camera position is adjusted relative to the position of the right and top boundary
        # todo_height
        renderView1.InteractionMode = '2D'
        renderView1.CameraPosition = [round(TRENCH_CENTER / 100e3) * 100e3, TOP-148468.75, 17717371.391353175]
        renderView1.CameraFocalPoint = [round(TRENCH_CENTER / 100e3) * 100e3, TOP-148468.75, 0.0]
        renderView1.CameraParallelScale = 179491.639356  # scale to a 2000 km domain
        
    else:
        raise NotImplementedError("plot_type PLOT_TYPE is not implemented")
    
    field, fieldLUT = set_viscosity_plot(sourceDisplay, 1e18, 1e24)
    scalarBar = set_viscosity_colorbar(renderView1, fieldLUT)
    if "PLOT_TYPE" in ["trench_centered", "orogen"]:
        # for this specific plot type, we want to add color bars later
        # hide colorbar
        sourceDisplay.SetScalarBarVisibility(renderView1, False)

    fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_viscosity_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_viscosity_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # plot temperature
    fieldLUT0 = fieldLUT
    field, fieldLUT = set_temperature_plot(sourceDisplay)
    HideScalarBarIfNotNeeded(fieldLUT0, renderView1) # hide previous colorbar
    scalarBar = set_temperature_colorbar(renderView1, fieldLUT)

    if ANIMATION:
        # turn on axis grid when making animation
        sourceDisplay.DataAxesGrid.GridAxesVisibility = 1
    
    if "PLOT_TYPE" in ["trench_centered", "orogen"]:
        # for this specific plot type, we want to add color bars later
        # hide colorbar
        sourceDisplay.SetScalarBarVisibility(renderView1, False)
    
    fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_temperature_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_temperature_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)
    
    # plot density
    fieldLUT0 = fieldLUT
    field, fieldLUT = set_density_plot(sourceDisplay)
    HideScalarBarIfNotNeeded(fieldLUT0, renderView1) # hide previous colorbar
    scalarBar = set_density_colorbar(renderView1, fieldLUT)
    if ANIMATION:
        # turn on axis grid when making animation
        sourceDisplay.DataAxesGrid.GridAxesVisibility = 1
    
    if "PLOT_TYPE" in ["trench_centered", "orogen"]:
        # for this specific plot type, we want to add color bars later
        # hide colorbar
        sourceDisplay.SetScalarBarVisibility(renderView1, False)

    fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_density_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_density_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)
    
    # plot composition indicators
    current_field = "composition_indicator"
    fieldLUT0 = fieldLUT
    ColorBy(sourceDisplay, ('POINTS', current_field, 'Magnitude'))
    fieldLUT = GetColorTransferFunction(current_field)
    HideScalarBarIfNotNeeded(fieldLUT0, renderView1) # hide previous colorbar
    if ANIMATION:
        # turn on axis grid when making animation
        sourceDisplay.DataAxesGrid.GridAxesVisibility = 1
    
    if "PLOT_TYPE" in ["trench_centered", "orogen"]:
        # for this specific plot type, we want to add color bars later
        # hide colorbar
        sourceDisplay.SetScalarBarVisibility(renderView1, False)
        renderView1.OrientationAxesVisibility = 0

        # additional inputs when particles are included
        if FOO01:
            # turn on the pin lines
            source_pin = possible_lookup_source("pin_lines_threshold1")
            sourceDisplayPin = Show(source_pin, renderView1, 'GeometryRepresentation')
            set_pin_line_plot(sourceDisplayPin)

            # turn on the active deforming region if the "active_deforming" field exists
            if source_has_field(_source, "active_deforming"):
                source_ad = possible_lookup_source("solution_active_deforming_threshold1")
                sourceDisplayAd = Show(source_ad, renderView1, 'GeometryRepresentation')
                set_active_deforming_plot(sourceDisplayAd)


    fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_%s_t%.4e.pdf" % (current_field, _time))
    fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_%s_t%.4e.png" % (current_field, _time))
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # plot fastscape outputs
    _sourceF = None
    _sourceFPG = None
    if INCLUDE_FASTSCAPE:
        source_name = "fastscape_transform1" 
        _sourceF = possible_lookup_source(source_name)
        source_name_pg = "fastscape_programmable1"
        _sourceFPG = possible_lookup_source(source_name_pg)

    if "PLOT_TYPE" == "full_domain":

        # adjust cammera to an orthoganal position
        layout_resolution = (1350, 704)

        if ANIMATION:
            # turn on axis grid when making animation
            sourceDisplay.DataAxesGrid.GridAxesVisibility = 1

        layout1 = GetLayout()
        layout1.SetSize((layout_resolution[0], layout_resolution[1]))

        # adjust camera setup
        # Camera position is adjusted relative to the position of the right and top boundary
        if np.isclose(RIGHT, 5500e3, rtol=1e-6) and np.isclose(TOP, 2000e3, rtol=1e-6):
            renderView1.InteractionMode = '2D'
            renderView1.CameraPosition = [9156148.523443608, 8267916.494211691, 6066660.785268369]
            renderView1.CameraFocalPoint = [2442318.849651574, 1554086.8204196477, -647168.8885236793]
            renderView1.CameraViewUp=[-0.4082482904638625, 0.816496580927726, -0.4082482904638639]
            renderView1.CameraParallelScale = 1698912.7130920088  # scale to the length
        else:
            raise NotImplementedError()

        # turn on viscosity plot for the main source
        # and the topography fro the fastscape source
        # TODO: use the new functions in base.py to set up plot and colorbar
        field, fieldLUT = set_viscosity_plot(sourceDisplay, 1e18, 1e24)
        scalarBar = set_viscosity_colorbar(renderView1, fieldLUT, position=[0.05, 0.05])

        if INCLUDE_FASTSCAPE: 
            sourceDisplayF = Show(_sourceF, renderView1, 'GeometryRepresentation')

            # set topography plot
            field, fieldLUT = set_fastscape_topography_plot(sourceDisplayF)
            scalarBar = set_fastscape_topography_colorbar(renderView1, fieldLUT, position=[0.6, 0.05])

            fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_topography_t%.4e.pdf" % (_time))
            fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_topography_t%.4e.png" % (_time))
            SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
            ExportView(fig_path, view=renderView1)
            print("Figure saved: %s" % fig_png_path)
            print("Figure saved: %s" % fig_path)

            # set erosion plot
            fieldLUT0 = fieldLUT
            rate_source = Calculator(Input=_sourceF)
            rate_source.ResultArrayName = "signed_sedimentation_rate"
            rate_source.Function = "-erosion_rate"
            Hide(_sourceF, renderView1)
            rate_display = Show(rate_source, renderView1, 'GeometryRepresentation')
            field, fieldLUT = set_fastscape_erosion_rate_plot(rate_display)
            scalarBar = set_fastscape_erosion_rate_colorbar(renderView1, fieldLUT, position=[0.6, 0.05])
            HideScalarBarIfNotNeeded(fieldLUT0, renderView1) # hide previous colorbar
            erosion_rate_contour = add_optional_fastscape_contour(
                _sourceF, "erosion_rate", 1e-3, renderView1,
                registration_name="erosion_rate_1e-3"
            )
            fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_erosion_rate_t%.4e.pdf" % (_time))
            fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_erosion_rate_t%.4e.png" % (_time))
            SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
            ExportView(fig_path, view=renderView1)
            remove_optional_fastscape_contour(erosion_rate_contour, renderView1)
            Hide(rate_source, renderView1)
            Delete(rate_source)
            Show(_sourceF, renderView1, 'GeometryRepresentation')
            print("Figure saved: %s" % fig_png_path)
            print("Figure saved: %s" % fig_path)
        
            # set total erosion plot
            fieldLUT0 = fieldLUT
            field, fieldLUT = set_fastscape_total_erosion_plot(sourceDisplayF)
            scalarBar = set_fastscape_total_erosion_colorbar(renderView1, fieldLUT, position=[0.6, 0.05])
            HideScalarBarIfNotNeeded(fieldLUT0, renderView1) # hide previous colorbar
            total_erosion_contour = add_optional_fastscape_contour(
                _sourceF, "total_erosion", 1e3, renderView1,
                registration_name="total_erosion_1e3"
            )
            fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_total_erosion_t%.4e.pdf" % (_time))
            fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_total_erosion_t%.4e.png" % (_time))
            SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
            ExportView(fig_path, view=renderView1)
            remove_optional_fastscape_contour(total_erosion_contour, renderView1)
            print("Figure saved: %s" % fig_png_path)
            print("Figure saved: %s" % fig_path)

            # Plot marine sediment thickness as negative signed deposition.
            Hide(_sourceF, renderView1)
            fieldLUT0 = fieldLUT
            HideScalarBarIfNotNeeded(fieldLUT0, renderView1)
            sourceDisplayFPG = Show(
                _sourceFPG, renderView1, 'GeometryRepresentation'
            )
            field, fieldLUT = set_fastscape_total_sedimentation_plot(
                sourceDisplayFPG
            )
            scalarBar = set_fastscape_total_sedimentation_colorbar(
                renderView1, fieldLUT, position=[0.6, 0.05]
            )
            sedimentation_contour = add_optional_fastscape_contour(
                _sourceFPG, "sedimentation", -1e3, renderView1,
                registration_name="sedimentation_minus_1e3"
            )
            fig_path = os.path.join(
                pv_output_dir,
                "PLOT_TYPE_total_sedimentation_t%.4e.pdf" % _time,
            )
            fig_png_path = os.path.join(
                pv_output_dir,
                "PLOT_TYPE_total_sedimentation_t%.4e.png" % _time,
            )
            SaveScreenshot(
                fig_png_path, renderView1, ImageResolution=layout_resolution
            )
            ExportView(fig_path, view=renderView1)
            remove_optional_fastscape_contour(sedimentation_contour, renderView1)
            print("Figure saved: %s" % fig_png_path)
            print("Figure saved: %s" % fig_path)

def thd_workflow(pv_output_dir, data_output_dir, steps, times):
    '''
    Workflow for the thd case
    Inputs:
        pv_output_dir - directory for figure
        data_output_dir - where the original case output locates
        steps - time steps
        times - corresponding times
    '''
    # placeholder
    pass

def add_composition_indicator(source_name):
    
    source = FindSource(source_name)

    # create a new 'Programmable Filter'
    programmableFilter1 = ProgrammableFilter(registrationName='Programmable_comp', Input=source)
    
    # Properties modified on programmableFilter1
    programmableFilter1.Script = """
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

composition_fields = ['gabbro', 'MORB', 'sediment', 'crust_upper', 'crust_lower']

input0 = self.GetInputDataObject(0, 0)
output.ShallowCopy(input0)

pd = input0.GetPointData()

# ---- composition arrays ----
arrays = []
for name in CONTINENT_COMPOSITIONS:
    arr = pd.GetArray(name)
    if arr is None:
        raise RuntimeError(f"Missing array: {name}")
    arrays.append(vtk_to_numpy(arr))

# ---- oceanic crust: sum up the value
oceanic_crust_compositions = OCEANIC_CRUST_COMPOSITIONS
arr = vtk_to_numpy(pd.GetArray(oceanic_crust_compositions[0]))
for name in oceanic_crust_compositions[1:]:
    arr += vtk_to_numpy(pd.GetArray(name))
arrays.append(arr)

# ---- sediment
arr = pd.GetArray('sediment')
arrays.append(vtk_to_numpy(arr))

data = np.vstack(arrays).T

# ---- background ----
sum_composition = np.sum(data, axis=1)
background = np.clip(1.0 - sum_composition, 0.0, 1.0)

# ---- split background by temperature ----
T = vtk_to_numpy(pd.GetArray("T"))
if T is None:
    raise RuntimeError("Missing temperature field: T")

cold_bg = background * (T < LITHOSPHERE_TEMPERATURE)
warm_bg = background * (T >= LITHOSPHERE_TEMPERATURE)

# ---- full classification matrix ----
data_with_bg = np.hstack([
    warm_bg[:, None],
    data,
    cold_bg[:, None]
])

# ---- argmax ----
indicator = np.argmax(data_with_bg, axis=1)

# ---- write output ----
vtk_arr = numpy_to_vtk(indicator.astype(np.int32))
vtk_arr.SetName("composition_indicator")
output.GetPointData().AddArray(vtk_arr)
"""

def setup_fastscape(source_i):
    """
    Setup paraview object for fastscape.
    With a Transform filter, it's rotated and translated to the surface of the model.
    Inputs:
    source_i - paraview source or source name
    """

    input_source = possible_lookup_source(source_i) 
    transform = Transform(registrationName="fastscape_transform1", Input=input_source)
    transform.Transform.Rotate = [-90.0, 0.0, 0.0]
    transform.Transform.Translate = [0.0, TOP, 0.0]

    transform_source = FindSource("fastscape_transform1")
    sedimentation = ProgrammableFilter(
        registrationName="fastscape_programmable1",
        Input=transform_source,
    )
    sedimentation.Script = """
Topography = inputs[0].PointData["topography"]
Basement = inputs[0].PointData["basement"]
Sedimentation = Basement - Topography

output.PointData.append(
    Sedimentation,
    "sedimentation"
)
"""

# For some reason, the SCM is not automatically loaded anymore
# and needs to be loaded manually.
PV_INTERFACE_PATH = os.environ["PV_INTERFACE_PATH"]
if PV_INTERFACE_PATH != "" and os.path.isdir(PV_INTERFACE_PATH):
    if not color_preset_exists("SCM_roma"):
        print("import files under: %s" % PV_INTERFACE_PATH)
        import_xmls(PV_INTERFACE_PATH)
    else:
        print("preset exists, skip importing color xml files.")

# check consistency in setup
if "PLOT_TYPE" == "orogen":
    if not FOO00:
        raise ValueError("Needs to turn on the pyvista flag for plotting\"orogen\"")

steps = GRAPHICAL_STEPS
times = GRAPHICAL_TIMES
timesteps = GRAPHICAL_TIME_STEPS
visualization_dir = "VISUALIZATION_DIRECTORY"
pv_output_dir = os.path.abspath(os.path.join("IMAGE_DIRECTORY", "pv_outputs"))

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
    

if int("DIMENSION") == 3:
    thd_workflow(pv_output_dir, visualization_dir, steps, times)
else:
    twod_workflow(pv_output_dir, visualization_dir, steps, times, timesteps)
