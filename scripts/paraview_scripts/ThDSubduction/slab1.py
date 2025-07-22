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


def add_trench_triangle(registration_name, geometry, cx, cy, cz, size):
    # get active source.
    trenchTrSource = ProgrammableSource(registrationName=registration_name)
    if geometry == "chunk":
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
    else:
        trenchTrSource.Script = """from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList

# === Set the center location of the triangle ===
cx, cy, cz = %.1f, %.1f, %.1f  # <-- CHANGE THIS TO WHERE YOU WANT THE MARKER

# === Triangle defined relative to the center ===
# This triangle lies in the XY plane, pointing downward
size = %.1f # overall size scale

p0 = [cx - size, cy,     cz]      # left base
p1 = [cx + size, cy,     cz]      # right base
p2 = [cx, cy,     cz - 2*size]  # tip (pointing down)

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

def plot_slice_center_viscosity(source_name, snapshot, pv_output_dir, _time):

    # Show the slab center plot and viscosities
    if "GEOMETRY" == "chunk":
        transform1 = FindSource("%s_transform_%05d" % (source_name, snapshot))
    else:
        transform1 = FindSource("%s_%05d" % (source_name, snapshot))
    SetActiveSource(transform1)
    renderView1 = GetActiveViewOrCreate('RenderView')
    transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')

    set_viscosity_plot(transform1Display, ETA_MIN, ETA_MAX)
    
    # Adjust the position of the point source and show related annotations.
    # Adjust glyph properties based on the specified parameters.
    sourceV = FindSource("%s_glyph_%05d" % (source_name, snapshot))
    sourceVDisplay = Show(sourceV, renderView1, 'GeometryRepresentation')
    if int("DIMENSION") == 3:
        ColorBy(sourceVDisplay, None)
    # sourceVDisplay.SetScalarBarVisibility(renderView1, True)
    pointSource1 = FindSource("PointSource_%s_glyph_%05d" % (source_name, snapshot))

    if False:
        # show the representative vector
        sourceVRE = FindSource("%s_glyph_%05d_representative" % (source_name, snapshot))
        sourceVREDisplay = Show(sourceVRE, renderView1, 'GeometryRepresentation')
        sourceVTXT = FindSource("%s_glyph_%05d_text" % (source_name, snapshot))
        sourceVTXTDisplay = Show(sourceVTXT, renderView1, 'GeometryRepresentation')
        sourceVTXTDisplay.Color = [0.0, 0.0, 0.0]

    scale_factor = 2.8e6
    n_sample_points = 2500
    point_source_center = [0.0, 0.0, 0.0]
    if "GEOMETRY" == "chunk":
        point_source_center = [0, 6.4e6, 0]
    elif "GEOMETRY" == "box":
        point_source_center = [4.65e6, 2.95e6, 0]
    else:
        raise NotImplementedError()
    adjust_glyph_properties("%s_glyph_%05d" % (source_name, snapshot), scale_factor, n_sample_points, point_source_center)


    # Show the original trench position
    sourceTrOrigTrian = FindSource("trench_orig_triangle")
    sourceTrOrigTrianDisplay = Show(sourceTrOrigTrian, renderView1, 'GeometryRepresentation')
    
    # Show the current trench position
    sourceTrTrian = FindSource("trench_triangle")
    sourceTrTrianDisplay = Show(sourceTrTrian, renderView1, 'GeometryRepresentation')

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
    fig_path = os.path.join(pv_output_dir, "slice_center_viscosity_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "slice_center_viscosity_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # hide objects
    # Hide(transform1, renderView1)
    # Hide(sourceV, renderView1)
    # Hide(pointSource1, renderView1)
    # Hide(sourceVRE, renderView1)
    # Hide(sourceVTXT, renderView1)
    # fieldLUT = GetColorTransferFunction("viscosity")
    # HideScalarBarIfNotNeeded(fieldLUT, renderView1)


def plot_slab_velocity_field(snapshot, _time, pv_output_dir):
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

    # Show the plane at 660 km
    transform_660 = FindSource("plane_660.0km%s" % trailer)
    SetActiveSource(transform_660)
    transform_660Display = Show(transform_660, renderView1, 'GeometryRepresentation')
    transform_660Display.SetRepresentationType('Feature Edges')
    transform_660Display.AmbientColor = [1.0, 0.3333333333333333, 0.0]
    transform_660Display.DiffuseColor = [1.0, 0.3333333333333333, 0.0]
    transform_660Display.LineWidth = 2.0

    # Show the slab surface
    transform_slab = FindSource("sp_lower_above_0.8_filtered_pe%s_%05d" % (trailer, snapshot))
    SetActiveSource(transform_slab)
    transform_slabDisplay = Show(transform_slab, renderView1, 'GeometryRepresentation')
    set_slab_volume_plot(transform_slabDisplay, 1000e3)
    transform_slabDisplay.SetScalarBarVisibility(renderView1, True)


    # Show the slice center surface
    source_center = FindSource("slice_center%s_%05d" % (trailer, snapshot))
    source_centerDisplay = Show(source_center, renderView1, 'GeometryRepresentation')
    ColorBy(source_centerDisplay, None)
    source_centerDisplay.Opacity = 0.3

    # Show the slice center glyph
    # And Adjust glyph properties based on the specified parameters.
    sourceV1 = FindSource("slice_center_glyph_%05d" % snapshot)
    sourceV1.GlyphMode = 'Uniform Spatial Distribution (Surface Sampling)'
    sourceV1Display = Show(sourceV1, renderView1, 'GeometryRepresentation')
    ColorBy(sourceV1Display, None)
    sourceV1Display.AmbientColor = [0.6666666666666666, 0.0, 1.0]
    sourceV1Display.DiffuseColor = [0.6666666666666666, 0.0, 1.0]

    scale_factor = 1e7
    n_sample_points = 500
    if "GEOMETRY" == "chunk":
        x_p, y_p, z_p = rotate_spherical_point_paraview_style(OUTER_RADIUS+250e3, np.pi/2.0, TRENCH_INI_DERIVED, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
        point_source_center = [x_p, y_p, z_p]
    elif "GEOMETRY" == "box":
        point_source_center = [TRENCH_INI_DERIVED, 0.0, OUTER_RADIUS+250e3]
    else:
        raise NotImplementedError()
    adjust_glyph_properties("slice_center_glyph_%05d" % snapshot, scale_factor, n_sample_points, point_source_center)

    # Show the slice at 200 km depth surface
    source_slice_200km = FindSource("slice_depth_200.0km%s_%05d" % (trailer, snapshot))
    source_slice_200kmDisplay = Show(source_slice_200km, renderView1, 'GeometryRepresentation')
    if "GEOMETRY" == "box":
        ColorBy(source_slice_200kmDisplay, None)
    source_slice_200kmDisplay.Opacity = 0.2

    # Show the slice at 200 km depth glyph
    # And Adjust glyph properties based on the specified parameters.
    sourceV2 = FindSource("slice_depth_200.0km_glyph_%05d" % snapshot)
    sourceV2.GlyphMode = 'Uniform Spatial Distribution (Surface Sampling)'
    sourceV2Display = Show(sourceV2, renderView1, 'GeometryRepresentation')
    if "GEOMETRY" == "box":
        ColorBy(sourceV2Display, None)
    sourceV2Display.AmbientColor = [0.0, 0.3333333333333333, 1.0]
    sourceV2Display.DiffuseColor = [0.0, 0.3333333333333333, 1.0]

    scale_factor = 1e7
    n_sample_points = 1000
    if "GEOMETRY" == "chunk":
        x_p, y_p, z_p = rotate_spherical_point_paraview_style(OUTER_RADIUS+250e3, np.pi/2.0, TRENCH_INI_DERIVED, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
        point_source_center = [x_p, y_p, z_p]
    elif "GEOMETRY" == "box":
        point_source_center = [TRENCH_INI_DERIVED, 0.0, OUTER_RADIUS+250e3]
    else:
        raise NotImplementedError()
    adjust_glyph_properties("slice_depth_200.0km_glyph_%05d" % snapshot, scale_factor, n_sample_points, point_source_center)
    
    # Show the slice at 200 km depth glyph representative
    if False:
        sourceV2_rep = FindSource("slice_depth_200.0km_glyph_%05d_representative" % snapshot)
        sourceV2_repDisplay = Show(sourceV2_rep, renderView1, 'GeometryRepresentation')

    # Show the original trench position
    sourceTrOrigTrian = FindSource("trench_orig_triangle")
    sourceTrOrigTrianDisplay = Show(sourceTrOrigTrian, renderView1, 'GeometryRepresentation')
    
    # Show the current trench position
    sourceTrTrian = FindSource("trench_triangle")
    sourceTrTrianDisplay = Show(sourceTrTrian, renderView1, 'GeometryRepresentation')

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
        renderView1.CameraPosition = [-3977427.3881134638, 9412059.879205279, 6047215.552680263]
        renderView1.CameraFocalPoint = [1731504.8595461769, 3105601.1729913107, -1755775.615996314]
        renderView1.CameraViewUp = [0.4314695504273366, 0.8294679274563145, -0.35470689924973053]
        renderView1.CameraParallelScale = 600000.0
    elif "GEOMETRY" == "box":
        renderView1.CameraPosition = [7901107.27312585, 5768057.049286575, 5496301.138370097]
        renderView1.CameraFocalPoint = [2242318.5206255154, -1901461.3695231928, 369527.98904717574]
        renderView1.CameraViewUp = [-0.3457156989670518, -0.3316734584620678, 0.8777661262771159]
        renderView1.CameraParallelScale = 600000.0

    # save figure
    fig_name = "3d_velocity_%.4e" % (float(_time))
    fig_path = os.path.join(pv_output_dir, "%s.png" % (fig_name))
    fig_pdf_path = os.path.join(pv_output_dir, "%s.pdf" % (fig_name))
    SaveScreenshot(fig_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_pdf_path, view=renderView1)
    print("Figure saved: %s" % fig_path)
    print("Figure saved: %s" % fig_pdf_path)

def thd_workflow(pv_output_dir, data_output_dir, steps, times):
    # load model boundary
    load_pyvista_source(data_output_dir, "model_boundary", None, file_type="vtu")
    load_pyvista_source(data_output_dir, "model_boundary_marker_points", None)
    load_pyvista_source(data_output_dir, "plane_660.0km", None, file_type="vtu")

    # add a position of the original trench
    if "GEOMETRY" == "chunk":
        x_tr_orig, y_tr_orig, z_tr_orig = rotate_spherical_point_paraview_style(OUTER_RADIUS+100e3, np.pi/2.0, TRENCH_INI_DERIVED, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
        add_trench_triangle("trench_orig_triangle", "GEOMETRY", x_tr_orig, y_tr_orig, z_tr_orig, 45e3)
        trench_orig = FindSource("trench_orig_triangle")
        trench_origDisplay = Show(trench_orig, renderView1, 'GeometryRepresentation')
        trench_origDisplay.AmbientColor = [1.0, 0.6666666666666666, 0.0]
        trench_origDisplay.DiffuseColor = [1.0, 0.6666666666666666, 0.0]
        Hide(trench_orig, renderView1)
    else:
        x_tr_orig, y_tr_orig, z_tr_orig = TRENCH_INI_DERIVED, 0.0, OUTER_RADIUS+100e3
        add_trench_triangle("trench_orig_triangle", "GEOMETRY", x_tr_orig, y_tr_orig, z_tr_orig, 45e3)
        trench_orig = FindSource("trench_orig_triangle")
        trench_origDisplay = Show(trench_orig, renderView1, 'GeometryRepresentation')
        trench_origDisplay.AmbientColor = [1.0, 0.6666666666666666, 0.0]
        trench_origDisplay.DiffuseColor = [1.0, 0.6666666666666666, 0.0]
        Hide(trench_orig, renderView1)

    # add a position of the current trench
    if "GEOMETRY" == "chunk":
        x_tr, y_tr, z_tr = rotate_spherical_point_paraview_style(OUTER_RADIUS+100e3, np.pi/2.0, TRENCH_CENTER, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
        add_trench_triangle("trench_triangle", "GEOMETRY", x_tr, y_tr, z_tr, 45e3)
        trench = FindSource("trench_triangle")
        trenchDisplay = Show(trench, renderView1, 'GeometryRepresentation')
        trenchDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0]
        trenchDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]
        Hide(trench, renderView1)
    else:
        x_tr, y_tr, z_tr = TRENCH_CENTER, 0.0, OUTER_RADIUS+100e3
        add_trench_triangle("trench_triangle", "GEOMETRY", x_tr, y_tr, z_tr, 45e3)
        trench = FindSource("trench_triangle")
        trenchDisplay = Show(trench, renderView1, 'GeometryRepresentation')
        trenchDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0]
        trenchDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]
        Hide(trench, renderView1)

    # loop every step to plot
    for i, step in enumerate(steps):
        snapshot = INITIAL_ADAPTIVE_REFINEMENT+step
        _time = times[i]

        # load slice center
        load_pyvista_source(data_output_dir, "slice_center_unbounded", snapshot, file_type="vtp", assign_field=True, add_glyph=True)
        load_pyvista_source(data_output_dir, "slice_center", snapshot, file_type="vtu", assign_field=True, add_glyph=True)

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
        # plot_slice_center_viscosity("slice_center_unbounded", snapshot, pv_output_dir, _time)
        
        # plot slab_velocity_field
        plot_slab_velocity_field(snapshot, _time, pv_output_dir)

def twod_workflow(pv_output_dir, data_output_dir, steps, times):
    '''
    Workflow for the twod case
    Inputs:
        pv_output_dir - directory for figure
        data_output_dir - where the original case output locates
        steps - time steps
        times - corresponding times
    '''
    # add a position of the original trench
    if "GEOMETRY" == "chunk":
        x_tr_orig, y_tr_orig, z_tr_orig = rotate_spherical_point_paraview_style(OUTER_RADIUS+100e3, np.pi/2.0, TRENCH_INI_DERIVED, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
        add_trench_triangle("trench_orig_triangle", "GEOMETRY", x_tr_orig, y_tr_orig, z_tr_orig, 45e3)
        trench_orig = FindSource("trench_orig_triangle")
        trench_origDisplay = Show(trench_orig, renderView1, 'GeometryRepresentation')
        trench_origDisplay.AmbientColor = [1.0, 0.6666666666666666, 0.0]
        trench_origDisplay.DiffuseColor = [1.0, 0.6666666666666666, 0.0]
        Hide(trench_orig, renderView1)
    else:
        x_tr_orig, y_tr_orig, z_tr_orig = TRENCH_INI_DERIVED, 0.0, OUTER_RADIUS+100e3
        add_trench_triangle("trench_orig_triangle", "GEOMETRY", x_tr_orig, y_tr_orig, z_tr_orig, 45e3)
        trench_orig = FindSource("trench_orig_triangle")
        trench_origDisplay = Show(trench_orig, renderView1, 'GeometryRepresentation')
        trench_origDisplay.AmbientColor = [1.0, 0.6666666666666666, 0.0]
        trench_origDisplay.DiffuseColor = [1.0, 0.6666666666666666, 0.0]
        Hide(trench_orig, renderView1)

    # add a position of the current trench
    if "GEOMETRY" == "chunk":
        x_tr, y_tr, z_tr = rotate_spherical_point_paraview_style(OUTER_RADIUS+100e3, np.pi/2.0, TRENCH_CENTER, rotate_deg=[0, 0, ROTATION_ANGLE], translate=[0, 0, 0])
        add_trench_triangle("trench_triangle", "GEOMETRY", x_tr, y_tr, z_tr, 45e3)
        trench = FindSource("trench_triangle")
        trenchDisplay = Show(trench, renderView1, 'GeometryRepresentation')
        trenchDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0]
        trenchDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]
        Hide(trench, renderView1)
    else:
        x_tr, y_tr, z_tr = TRENCH_CENTER, 0.0, OUTER_RADIUS+100e3
        add_trench_triangle("trench_triangle", "GEOMETRY", x_tr, y_tr, z_tr, 45e3)
        trench = FindSource("trench_triangle")
        trenchDisplay = Show(trench, renderView1, 'GeometryRepresentation')
        trenchDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0]
        trenchDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]
        Hide(trench, renderView1)
        
    for i, step in enumerate(steps):
        snapshot = INITIAL_ADAPTIVE_REFINEMENT+step
        _time = times[i]
    
        # add source
        filein = os.path.join(data_output_dir, "solution", "solution-%05d.pvtu" %snapshot) 

        source_name = 'solution' 
        XMLPartitionedUnstructuredGridReader(registrationName=source_name, FileName=[filein])
        
        # add rotation
        if "GEOMETRY" == "chunk":
            registration_name_transform = '%s_transform_%05d' % (source_name, snapshot)
            solution = FindSource(source_name)
            transform = Transform(registrationName=registration_name_transform, Input=solution)
            transform.Transform = 'Transform'
            transform.Transform.Translate = [0.0, 0.0, 0.0]  # center of rotation
            transform.Transform.Rotate = [0.0, 0.0, ROTATION_ANGLE]  # angle of rotation
            Hide3DWidgets()
        pass

        registration_name_glyph = '%s_glyph_%05d' % (source_name, snapshot)
        if "GEOMETRY" == "chunk":
            add_glyph1("%s_transform_%05d" % (source_name, snapshot), "velocity", 1e6, registrationName=registration_name_glyph)
        else:
            add_glyph1("%s_%05d" % (source_name, snapshot), "velocity", 1e6, registrationName=registration_name_glyph)

        # plot slice
        plot_slice_center_viscosity(source_name, snapshot, pv_output_dir, _time)


steps = GRAPHICAL_STEPS
times = GRAPHICAL_TIMES
data_output_dir = "DATA_OUTPUT_DIR"
pv_output_dir = os.path.abspath(os.path.join("DATA_OUTPUT_DIR", "..", "img", "pv_outputs"))

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
    

if int("DIMENSION") == 3:
    thd_workflow(pv_output_dir, data_output_dir, steps, times)
else:
    twod_workflow(pv_output_dir, data_output_dir, steps, times)