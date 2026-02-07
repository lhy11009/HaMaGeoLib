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
    fieldLUT.ApplyPreset("bilbao", True)
    return field, fieldLUT

def set_viscosity_colorbar(renderView, fieldLUT):
    '''
    set the viscosity colorbar
    Inputs:
        renderView - paraview object connect to visualization
        fieldLUT - color transfer function for viscosity
    ''' 
    scalarBar = GetScalarBar(fieldLUT, renderView)
    scalarBar.Visibility = 1
    scalarBar.Title = 'Viscosity'
    scalarBar.ComponentTitle = '[Pa·s]'
    scalarBar.Orientation = 'Horizontal'
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
    fieldLUT.ApplyPreset("lapaz", True)
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
    fieldLUT.ApplyPreset("batlow", True)
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


def set_data_axes_grid(renderView1):
    '''
    toggle the data axes for display geologic domain
    Inputs:
        renderView1 - paraview object connect to visualization
    '''
    renderView1.AxesGrid.Visibility = 1

    if (pv_major > 5) or (pv_major == 5 and pv_minor >= 11):
        # ParaView ≥ 5.11: axis scaling is supported
        renderView1.AxesGrid.XAxisScale = 1.0e-3
        renderView1.AxesGrid.YAxisScale = 1.0e-3
        renderView1.AxesGrid.ZAxisScale = 1.0e-3

        renderView1.AxesGrid.XTitle = 'X [km]'
        renderView1.AxesGrid.YTitle = 'Y [km]'
        renderView1.AxesGrid.ZTitle = 'Z [km]'
    else:
        # ParaView 5.10 and older: no axis scaling support
        # Keep units honest
        renderView1.AxesGrid.XTitle = 'X [m]'
        renderView1.AxesGrid.YTitle = 'Y [m]'
        renderView1.AxesGrid.ZTitle = 'Z [m]'


def twod_workflow(pv_output_dir, data_output_dir, steps, times):
    '''
    Workflow for the twod case
    Inputs:
        pv_output_dir - directory for figure
        data_output_dir - where the original case output locates
        steps - time steps
        times - corresponding times
    '''
    for i, step in enumerate(steps):
        if PP_INITIAL_REFINEMENT:
            snapshot = INITIAL_ADAPTIVE_REFINEMENT+step
        else:
            snapshot = step
        _time = times[i]
    
        # add source
        filein = os.path.join(data_output_dir, "solution", "solution-%05d.pvtu" %snapshot) 

        XMLPartitionedUnstructuredGridReader(registrationName='solution_%05d' % snapshot, FileName=[filein])

        plot_full_domain("solution-%05d" % snapshot, _time, pv_output_dir)


def plot_full_domain(source_name, _time, pv_output_dir):
        
    _source = FindSource(source_name)
    sourceDisplay = Show(_source, renderView1, 'GeometryRepresentation')

    # todo_collision
    # plot viscosity    
    layout_resolution = (1350, 704)

    field, fieldLUT = set_viscosity_plot(sourceDisplay, 1e18, 1e24)
    scalarBar = set_viscosity_colorbar(renderView1, fieldLUT)
    if ANIMATION:
        # sourceDisplay.DataAxesGrid.GridAxesVisibility = 1 # set axes grid to visible
        set_data_axes_grid(renderView1)


    layout1 = GetLayout()
    layout1.SetSize((layout_resolution[0], layout_resolution[1]))

    # adjust camera setup
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [4254403.917056384, 715564.0158183385, 17717371.391353175]
    renderView1.CameraFocalPoint = [4254403.917056384, 715564.0158183385, 0.0]
    renderView1.CameraParallelScale = 2588447.7843194483

    fig_path = os.path.join(pv_output_dir, "full_domain_viscosity_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "full_domain_viscosity_t%.4e.png" % _time)
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
        # sourceDisplay.DataAxesGrid.GridAxesVisibility = 1 # set axes grid to visible
        set_data_axes_grid(renderView1)
    
    fig_path = os.path.join(pv_output_dir, "full_domain_temperature_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "full_domain_temperature_t%.4e.png" % _time)
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
        # sourceDisplay.DataAxesGrid.GridAxesVisibility = 1 # set axes grid to visible
        set_data_axes_grid(renderView1)

    fig_path = os.path.join(pv_output_dir, "full_domain_density_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "full_domain_density_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # hide plot and colorbar
    # Hide(_source, renderView1)
    # HideScalarBarIfNotNeeded(fieldLUT, renderView1) # hide previous colorbar



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


steps = GRAPHICAL_STEPS
times = GRAPHICAL_TIMES
data_output_dir = "OUTPUT_DIRECTORY"
pv_output_dir = os.path.abspath(os.path.join("IMAGE_DIRECTORY", "pv_outputs"))

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
    

if int("DIMENSION") == 3:
    thd_workflow(pv_output_dir, data_output_dir, steps, times)
else:
    twod_workflow(pv_output_dir, data_output_dir, steps, times)