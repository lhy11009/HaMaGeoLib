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

    set_viscosity_plot(sourceDisplay, 1e18, 1e24)
    layout1 = GetLayout()
    layout1.SetSize((layout_resolution[0], layout_resolution[1]))

    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [4350000.0, 1450918.5, 17717371.391353175]
    renderView1.CameraFocalPoint = [4350000.0, 1450918.5, 0.0]
    renderView1.CameraParallelScale = 3789746.4010221055

    fig_path = os.path.join(pv_output_dir, "full_domain_viscosity_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "full_domain_viscosity_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # plot temperature
    set_temperature_plot(sourceDisplay)
    
    fig_path = os.path.join(pv_output_dir, "full_domain_temperature_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "full_domain_temperature_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)
    
    # plot density
    set_density_plot(sourceDisplay)

    fig_path = os.path.join(pv_output_dir, "full_domain_density_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "full_domain_density_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
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