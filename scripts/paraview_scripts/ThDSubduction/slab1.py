#### import the simple module from the paraview
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

#### Utility functions
def rotate_spherical_point_paraview_style(r, theta, phi, rotate_deg=[0, 0, 0], translate=[0, 0, 0]):
    """
    Rotate a point in spherical coordinates using ParaView-style rotation (ZYX Euler order).
    
    Parameters:
        r (float): Radius
        theta (float): Inclination from z-axis (in radians)
        phi (float): Azimuth from x-axis in xy-plane (in radians)
        rotate_deg (list of 3): Rotation angles around X, Y, Z axes (in degrees)
        translate (list of 3): Translation to apply before rotation (center of rotation)
    
    Returns:
        rotated_cartesian (np.array): [x', y', z']
        rotated_spherical (tuple): (r', theta', phi') in radians
    """

    # Step 1: Convert spherical to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    point = np.array([x, y, z])

    # Step 2: Translate to origin (if needed)
    point_shifted = point - np.array(translate)

    # Step 3: Apply rotation using ZYX Euler angles (ParaView-style)
    rot = R.from_euler('XYZ', rotate_deg, degrees=True)
    rotated = rot.apply(point_shifted)

    # Step 4: Translate back (if needed)
    rotated += np.array(translate)

    # Step 5: Convert back to spherical
    x_, y_, z_ = rotated

    return x_, y_, z_

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

def set_slab_volume_plot(sourceDisplay, max_depth, **kwargs):
    '''
    set the viscosity plot
    Inputs:
        eta_min (float): minimum viscosity
        eta_max (float): maximum viscosity
    '''
    opacity = kwargs.get("opacity", 1.0)
    field = "radius"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, OUTER_RADIUS-max_depth, OUTER_RADIUS)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("imola", True)
    sourceDisplay.Opacity = opacity


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
    glyph1.OrientationArray = ['POINTS', field]
    glyph1.ScaleArray = ['POINTS', field]
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

def load_pyvista_source(data_output_dir, source_name, snapshot, **kwargs):

    # options
    file_type = kwargs.get("file_type", "vtp")
    add_glyph = kwargs.get("add_glyph", False)
    assign_field = kwargs.get("assign_field", False)

    # load source file
    if file_type == "vtp":
        READER = XMLPolyDataReader
    elif file_type == "vtu":
        READER = XMLUnstructuredGridReader
    else:
        raise NotImplementedError("file_type needs to be vtp, vtu")

    if snapshot is None:
        filepath = '%s/../pyvista_outputs/%s.%s' % (data_output_dir, source_name, file_type)
        registration_name = source_name
    else:
        filepath = '%s/../pyvista_outputs/%05d/%s_%05d.%s' % (data_output_dir, snapshot, source_name, snapshot, file_type)
        registration_name = '%s_%05d' % (source_name, snapshot)
    if not os.path.isfile(filepath):
        raise FileNotFoundError("File %s is not found" % filepath)

    source = READER(registrationName=registration_name, FileName=[filepath])

    if assign_field:
        source.PointArrayStatus = ['velocity', 'p', 'T', 'sp_upper', 'sp_lower', 'density', 'viscosity',\
            'dislocation_viscosity', 'diffusion_viscosity', 'peierls_viscosity', 'strain_rate', 'velocity_slice', "radius"]

    # add rotation
    if "GEOMETRY" == "chunk":
        if snapshot is None:
            registration_name_transform = '%s_transform' % (source_name)
        else:
            registration_name_transform = '%s_transform_%05d' % (source_name, snapshot)
        solutionpvd = FindSource(registration_name)
        transform = Transform(registrationName=registration_name_transform, Input=solutionpvd)
        transform.Transform = 'Transform'
        transform.Transform.Translate = [0.0, 0.0, 0.0]  # center of rotation
        transform.Transform.Rotate = [0.0, 0.0, ROTATION_ANGLE]  # angle of rotation
        Hide3DWidgets()

    # add glyph 
    if add_glyph:
        if snapshot is None:
            registration_name_glyph = '%s_glyph' % (source_name)
        else:
            registration_name_glyph = '%s_glyph_%05d' % (source_name, snapshot)
        if "GEOMETRY" == "chunk":
            add_glyph1("%s_transform_%05d" % (source_name, snapshot), "velocity_slice", 1e6, registrationName=registration_name_glyph)
        else:
            add_glyph1("%s_%05d" % (source_name, snapshot), "velocity_slice", 1e6, registrationName=registration_name_glyph)


def add_trench_triangle(registration_name, cx, cy, cz, size):
    # get active source.
    trenchTrSource = ProgrammableSource(registrationName=registration_name)
    trenchTrSource.Script = """from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList

# === Set the center location of the triangle ===
cx, cy, cz = %.1f, %.1f, %.1f  # <-- CHANGE THIS TO WHERE YOU WANT THE MARKER

# === Triangle defined relative to the center ===
# This triangle lies in the XY plane, pointing downward
size = %.1f # overall size scale

p0 = [cx - size, cy,     cz]      # left base
p1 = [cx + size, cy,     cz]      # right base
p2 = [cx,         cy - size*2, cz]  # tip (pointing down)

# === Build triangle ===
points = vtkPoints()
points.InsertNextPoint(p0)
points.InsertNextPoint(p1)
points.InsertNextPoint(p2)

triangle = vtkIdList()
triangle.InsertNextId(0)
triangle.InsertNextId(1)
triangle.InsertNextId(2)

triangles = vtkCellArray()
triangles.InsertNextCell(triangle)

output.SetPoints(points)
output.SetPolys(triangles)""" % (cx, cy, cz, size)
    trenchTrSource.ScriptRequestInformation = ''
    trenchTrSource.PythonPath = ''


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


def plot_slab_velocity_field(snapshot, pv_output_dir):
    # get the renderView
    renderView1 = GetActiveViewOrCreate('RenderView')

    # trailer of source name
    if "GEOMETRY" == "chunk":
        trailer = "_transform"
    else:
        trailer = ""
    
    # Show the model boundary
    transform_bd = FindSource("model_boundary%s" % trailer)
    SetActiveSource(transform_bd)
    transform_bdDisplay = Show(transform_bd, renderView1, 'GeometryRepresentation')
    transform_bdDisplay.SetRepresentationType('Feature Edges')

    # Show the slab surface
    transform_slab = FindSource("sp_lower_above_0.8_filtered_pe%s_%05d" % (trailer, snapshot))
    SetActiveSource(transform_slab)
    transform_slabDisplay = Show(transform_slab, renderView1, 'GeometryRepresentation')
    set_slab_volume_plot(transform_slabDisplay, 1000e3)

    # Show the slice center glyph
    # And Adjust glyph properties based on the specified parameters.
    sourceV1 = FindSource("slice_center_glyph_%05d" % snapshot)
    sourceV1.GlyphMode = 'Uniform Spatial Distribution (Surface Sampling)'
    sourceV1Display = Show(sourceV1, renderView1, 'GeometryRepresentation')
    ColorBy(sourceV1Display, None)
    sourceV1Display.AmbientColor = [0.6666666666666666, 0.0, 1.0]
    sourceV1Display.DiffuseColor = [0.6666666666666666, 0.0, 1.0]

    scale_factor = 5e6
    n_sample_points = 1000
    if "GEOMETRY" == "chunk":
        point_source_center = [0, 6.4e6, 0]
    elif "GEOMETRY" == "box":
        point_source_center = [4.65e6, 2.95e6, 0]
    else:
        raise NotImplementedError()
    adjust_glyph_properties("slice_center_glyph_%05d" % snapshot, scale_factor, n_sample_points, point_source_center)

    # Show the slice at 200 km depth glyph
    # And Adjust glyph properties based on the specified parameters.
    sourceV2 = FindSource("slice_depth_200.0km_glyph_%05d" % snapshot)
    sourceV2.GlyphMode = 'Uniform Spatial Distribution (Surface Sampling)'
    sourceV2Display = Show(sourceV2, renderView1, 'GeometryRepresentation')
    ColorBy(sourceV2Display, None)
    sourceV2Display.AmbientColor = [0.0, 0.3333333333333333, 1.0]
    sourceV2Display.DiffuseColor = [0.0, 0.3333333333333333, 1.0]

    scale_factor = 5e6
    n_sample_points = 2000
    if "GEOMETRY" == "chunk":
        point_source_center = [0, 6.4e6, 0]
    elif "GEOMETRY" == "box":
        point_source_center = [4.65e6, 2.95e6, 0]
    else:
        raise NotImplementedError()
    adjust_glyph_properties("slice_depth_200.0km_glyph_%05d" % snapshot, scale_factor, n_sample_points, point_source_center)

    # Show the original trench position
    sourceTrOrigTrian = FindSource("trench_orig_triangle")
    sourceTrOrigTrianDisplay = Show(sourceTrOrigTrian, renderView1, 'GeometryRepresentation')

    # Show the trench position
    sourceTr = FindSource("trench%s_%05d" % (trailer, snapshot))
    sourceTrDisplay = Show(sourceTr, renderView1, 'GeometryRepresentation')
    sourceTrDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0]
    sourceTrDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]

    # Configure layout and camera settings based on geometry.
    layout_resolution = (1350, 704)
    layout1 = GetLayout()
    layout1.SetSize((layout_resolution[0], layout_resolution[1]))
    # renderView1.InteractionMode = '2D'
    if "GEOMETRY" == "chunk":
        renderView1.CameraPosition = [-4836584.013744108, 8089634.7649268415, 7308574.720701834]
        renderView1.CameraFocalPoint = [1528057.7205373822, 3628014.9415879566, -1225784.3147636713]
        renderView1.CameraViewUp = [0.4547624702546983, 0.8822092047301953, -0.1220574239330037]
        renderView1.CameraParallelScale = 600000.0
    elif "GEOMETRY" == "box":
        renderView1.CameraPosition = [10420037.17156642, 5096774.770188813, 6264706.593871739]
        renderView1.CameraFocalPoint = [954463.8330784661, -1150881.9263552597, -281716.0365199686]
        renderView1.CameraViewUp = [-0.3922048999477372, -0.31237589777368785, 0.8652147796628696]
        renderView1.CameraParallelScale = 600000.0

    # hide objects
    # Hide(transform_bd, renderView1)



steps = GRAPHICAL_STEPS
times = GRAPHICAL_TIMES
data_output_dir = "DATA_OUTPUT_DIR"
pv_output_dir = os.path.abspath(os.path.join("DATA_OUTPUT_DIR", "..", "img", "pv_outputs"))

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
    
# load model boundary
load_pyvista_source(data_output_dir, "model_boundary", None, file_type="vtu")
load_pyvista_source(data_output_dir, "model_boundary_marker_points", None)
# load_pyvista_source(data_output_dir, "plane_660.0km", None, file_type="vtp")


# add a position of the original trench
if "GEOMETRY" == "chunk":
    x_tr_orig, y_tr_orig, z_tr_orig = rotate_spherical_point_paraview_style(OUTER_RADIUS+150e3, np.pi/2.0, THETA_REF_TRENCH, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
    add_trench_triangle("trench_orig_triangle", x_tr_orig, y_tr_orig, z_tr_orig, 60e3)
    trench_orig = FindSource("trench_orig_triangle")
    trench_origDisplay = Show(trench_orig, renderView1, 'GeometryRepresentation')
    trench_origDisplay.AmbientColor = [1.0, 0.6666666666666666, 0.0]
    trench_origDisplay.DiffuseColor = [1.0, 0.6666666666666666, 0.0]
    Hide(trench_orig, renderView1)
else:
    pass

# add a position of the current trench
if "GEOMETRY" == "chunk":
    x_tr, y_tr, z_tr = rotate_spherical_point_paraview_style(OUTER_RADIUS+150e3, np.pi/2.0, TRENCH_CENTER, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
    add_trench_triangle("trench_triangle", x_tr_orig, y_tr_orig, z_tr_orig, 60e3)
    trench = FindSource("trench_triangle")
    trenchDisplay = Show(trench, renderView1, 'GeometryRepresentation')
    trenchDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0]
    trenchDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]
    Hide(trench, renderView1)
else:
    pass

# loop every step to plot
for i, step in enumerate(steps):
    snapshot = INITIAL_ADAPTIVE_REFINEMENT+step


    # load slice center
    load_pyvista_source(data_output_dir, "slice_center", snapshot, assign_field=True, add_glyph=True)

    # load slice at 200 km depth
    if "GEOMETRY" == "chunk":
        slice_depth_file_type = "vtu"
    else:
        slice_depth_file_type = "vtp"

    load_pyvista_source(data_output_dir, "slice_depth_200.0km", snapshot, file_type=slice_depth_file_type, assign_field=True, add_glyph=True)

    # load subducting plate 
    load_pyvista_source(data_output_dir, "sp_lower_above_0.8_filtered_pe", snapshot, file_type="vtu", assign_field=True)
    
    # load slab surfaces
    load_pyvista_source(data_output_dir, "sp_upper_surface", snapshot, file_type="vtp", assign_field=True)
    load_pyvista_source(data_output_dir, "sp_lower_surface", snapshot, file_type="vtp", assign_field=True)
    
    # load trench position
    load_pyvista_source(data_output_dir, "trench", snapshot, file_type="vtp")

    # plot slice center viscosity
    # plot_slice_center_viscosity(snapshot, pv_output_dir)
    
    # plot slab_velocity_field
    plot_slab_velocity_field(snapshot, pv_output_dir)