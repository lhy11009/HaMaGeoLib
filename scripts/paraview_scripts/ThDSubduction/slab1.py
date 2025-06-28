#### import the simple module from the paraview
import os
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

#### Utility functions
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

def adjust_glyph_properties(registrationName, scale_factor, n_value, point_center, **kwargs):
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

def load_center_slice(data_output_dir, snapshot):
    # load slice center
    slice_center_filepath = '%s/../pyvista_outputs/%05d/slice_center_%05d.vtp' % (data_output_dir, snapshot, snapshot)
    assert(os.path.isfile(slice_center_filepath))
    slice_center_vtp = XMLPolyDataReader(registrationName='slice_center_%05d.vtp' % snapshot, FileName=[slice_center_filepath])
    slice_center_vtp.PointArrayStatus = ['velocity', 'p', 'T', 'sp_upper', 'sp_lower', 'density', 'viscosity', 'dislocation_viscosity', 'diffusion_viscosity', 'peierls_viscosity', 'strain_rate', 'velocity_slice']

    # add rotation
    solutionpvd = FindSource('slice_center_%05d.vtp' % snapshot)
    transform = Transform(registrationName="slice_center_transform_%05d" % snapshot, Input=solutionpvd)
    transform.Transform = 'Transform'
    transform.Transform.Translate = [0.0, 0.0, 0.0]  # center of rotation
    transform.Transform.Rotate = [0.0, 0.0, ROTATION_ANGLE]  # angle of rotation
    Hide3DWidgets()

    # add glyph 
    add_glyph1("slice_center_transform_%05d" % snapshot, "velocity_slice", 1e6, registrationName="slice_center_glyph_%05d" % snapshot)

def plot_slice_center_viscosity(snapshot, pv_output_dir):

    # Show the slab center plot and viscosities
    transform1 = FindSource("slice_center_transform_%05d" % snapshot)
    SetActiveSource(transform1)
    renderView1 = GetActiveViewOrCreate('RenderView')
    transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')

    set_viscosity_plot(transform1Display, ETA_MIN, ETA_MAX)
    
    # Adjust the position of the point source and show related annotations.
    sourceV = FindSource("slice_center_glyph_%05d" % snapshot)
    sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
    # sourceVDisplay.SetScalarBarVisibility(renderView1, True)
    pointSource1 = FindSource("PointSource_slice_center_glyph_%05d" % snapshot)
    if "GEOMETRY" == "chunk":
        pointSource1.Center = [0, 6.7e6, 0]
    sourceVRE = FindSource("slice_center_glyph_%05d_representative" % snapshot)
    sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation')
    sourceVTXT = FindSource("slice_center_glyph_%05d_text" % snapshot)
    sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
    sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]

    # Adjust glyph properties based on the specified parameters.
    scale_factor = 1e6
    n_sample_points = 20000
    point_source_center = [0.0, 0.0, 0.0]
    if "GEOMETRY" == "chunk":
        point_source_center = [0, 6.4e6, 0]
    elif "GEOMETRY" == "box":
        point_source_center = [4.65e6, 2.95e6, 0]
    else:
        raise NotImplementedError()
    adjust_glyph_properties("slice_center_glyph_%05d" % snapshot, scale_factor, n_sample_points, point_source_center)

    # Configure layout and camera settings based on geometry.
    layout_resolution = (1350, 704)
    layout1 = GetLayout()
    layout1.SetSize((layout_resolution[0], layout_resolution[1]))
    renderView1.InteractionMode = '2D'
    if "GEOMETRY" == "chunk":
        renderView1.CameraPosition = [-74708.2999944719, 5867664.065060813, 24790239.31741349]
        renderView1.CameraFocalPoint = [-74708.2999944719, 5867664.065060813, 0.0]
        renderView1.CameraParallelScale = 651407.1273990012
    elif "GEOMETRY" == "box":
        renderView1.CameraPosition = [4700895.868280185, 2538916.5897593317, 15340954.822755022]
        renderView1.CameraFocalPoint = [4700895.868280185, 2538916.5897593317, 0.0]
        renderView1.CameraParallelScale = 487763.78047352127

    # save figure
    fig_path = os.path.join(pv_output_dir, "slice_center_viscosity_t%.4e.pdf" % times[i])
    fig_png_path = os.path.join(pv_output_dir, "slice_center_viscosity_t%.4e.png" % times[i])
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)

    # hide objects
    Hide(transform1, renderView1)
    Hide(sourceV, renderView1)
    Hide(pointSource1, renderView1)
    Hide(sourceVRE, renderView1)
    Hide(sourceVTXT, renderView1)
    fieldLUT = GetColorTransferFunction("viscosity")
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)



steps = GRAPHICAL_STEPS
times = GRAPHICAL_TIMES
data_output_dir = "DATA_OUTPUT_DIR"
pv_output_dir = os.path.abspath(os.path.join("DATA_OUTPUT_DIR", "..", "img", "pv_outputs"))

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# load model boundary
model_boundary_filepath = '%s/../pyvista_outputs/model_boundary.vtp' % (data_output_dir)
assert(os.path.isfile(model_boundary_filepath))
model_boundary_vtp = XMLPolyDataReader(registrationName='model_boundary.vtp', FileName=[model_boundary_filepath])
model_boundary_Display = Show(model_boundary_vtp, renderView1, 'GeometryRepresentation')
model_boundary_Display.SetRepresentationType('Feature Edges')
Hide(model_boundary_vtp, renderView1)

# loop every step to plot
for i, step in enumerate(steps):
    snapshot = INITIAL_ADAPTIVE_REFINEMENT+step

    # load slice center
    load_center_slice(data_output_dir, snapshot)

    # plot slice center viscosity
    plot_slice_center_viscosity(snapshot, pv_output_dir)