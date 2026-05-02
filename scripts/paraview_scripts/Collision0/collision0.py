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
    fieldLUT.ApplyPreset("bilbao", True)
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

        XMLPartitionedUnstructuredGridReader(registrationName='solution-%05d' % snapshot, FileName=[filein])

        plot_twod_basic("solution-%05d" % snapshot, _time, pv_output_dir)

def plot_twod_basic(source_name, _time, pv_output_dir):
        
    _source = FindSource(source_name)

    # Add indicator field
    # Set color table with discrete categories;
    # Define categories (value -> label)
    # Define colors (flattened list)
    lut = GetColorTransferFunction("composition_indicator")
    add_composition_indicator(source_name)
    programmable_source = FindSource("Programmable_comp")
    programmable_sourceDisplay = Show(programmable_source, renderView1, 'GeometryRepresentation')

    lut.InterpretValuesAsCategories = 1
    lut.AnnotationsInitialized = 1

    lut.Annotations = [
        "0", "Asthenosphere",
        "1", "Upper crust",
        "2", "Lower crust",
        "3", "Oceanic crust",
        "4", "Sea floor sediment",
        "5", "Lithosphere",
        "6", "Phase 6",
        "7", "Phase 7",
    ]

    lut.IndexedColors = [
        0.4039, 0.8000, 0.9294,
        0.8118, 0.8157, 0.8235,
        0.9294, 0.4000, 0.4667,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.1451, 0.5373, 0.2588,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]

    programmable_sourceDisplay.ColorArrayName = ["POINTS", "composition_indicator"]
    programmable_sourceDisplay.LookupTable = lut

    Hide(programmable_source)

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

    if "PLOT_TYPE" == "full_domain":
        sourceDisplay = Show(_source, renderView1, 'GeometryRepresentation')

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
        sourceDisplay = Show(_source, renderView1, 'GeometryRepresentation')

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
        hide_everything()

        # todo_topo
        programmable_source = FindSource("Programmable_comp")
        sourceDisplay = Show(programmable_source, renderView1, 'GeometryRepresentation')
       
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
    if "PLOT_TYPE" == "trench_centered":
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
    
    if "PLOT_TYPE" == "trench_centered":
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
    
    if "PLOT_TYPE" == "trench_centered":
        # for this specific plot type, we want to add color bars later
        # hide colorbar
        sourceDisplay.SetScalarBarVisibility(renderView1, False)

    fig_path = os.path.join(pv_output_dir, "PLOT_TYPE_density_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "PLOT_TYPE_density_t%.4e.png" % _time)
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