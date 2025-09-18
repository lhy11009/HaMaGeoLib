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

def set_metastable_plot(sourceDisplay):
    '''
    set the viscosity plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    field = "metastable"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, 0.0, 1.0)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("Viridis (matplotlib)", True)

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

def set_non_adiabatic_pressure_plot_slab(sourceDisplay):
    '''
    set the temperature plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    field = "nonadiabatic_pressure"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    rescale_transfer_function_combined(field, -1e9, 1e9)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("turku", True)

# todo
def set_non_adiabatic_pressure_plot_mantle(sourceDisplay):
    '''
    set the temperature plot
    Inputs:
        sourceDisplay - source of the display (renderview)
    '''
    field = "nonadiabatic_pressure"
    ColorBy(sourceDisplay, ('POINTS', field, 'Magnitude'))
    da_range = DA_RANGE
    rescale_transfer_function_combined(field, da_range[0], da_range[1])
    # rescale_transfer_function_combined(field, -1e8, 1e8)
    fieldLUT = GetColorTransferFunction(field)
    fieldLUT.ApplyPreset("turku", True)

def set_slab_volume_plot(sourceDisplay, max_depth, **kwargs):
    '''
    set the viscosity plot
    Inputs:
        sourceDisplay - source of the display (renderview)
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
    GlyphMode = kwargs.get("GlyphMode", "Uniform Spatial Distribution (Surface Sampling)")

    assert(type(n_value) == int)
    assert(type(point_center) == list and len(point_center) == 3)
    representative_value = kwargs.get("representative_value", 0.05)
    
    glyph1 = FindSource(registrationName)
    glyph1.ScaleFactor = scale_factor

    if GlyphMode == "Uniform Spatial Distribution (Surface Sampling)":
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
        field_list = ['velocity', 'p', 'T', 'sp_upper', 'sp_lower', 'density', 'viscosity',\
            'dislocation_viscosity', 'diffusion_viscosity', 'peierls_viscosity', 'strain_rate', 'velocity_slice', "radius"]
        if HAS_DYNAMIC_PRESSURE:
            field_list.append('nonadiabatic_pressure')
        if "MODEL_TYPE" == "mow":
            field_list.append('metastable')
        source.PointArrayStatus = field_list

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
        if int("DIMENSION") == 3:
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
        else:
            trenchTrSource.Script = """from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList

# === Set the center location of the triangle ===
cx, cy, cz = %.1f, %.1f, %.1f  # <-- CHANGE THIS TO WHERE YOU WANT THE MARKER

# === Triangle defined relative to the center ===
# This triangle lies in the XY plane, pointing downward
size = %.1f # overall size scale

p0 = [cx - size, cz,     cy]      # left base
p1 = [cx + size, cz,     cy]      # right base
p2 = [cx, cz - 2*size, cy]  # tip (pointing down)

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


def add_eq_410_condition(source):
    programmableFilter1 = ProgrammableFilter(registrationName="programmable_eq", Input=source)
    programmableFilter1.Script = \
"""
import numpy as np 
T = inputs[0].PointData["T"]
p = inputs[0].PointData["p"]
p_eq = (T - 1780.0)*2e6 + 1.34829e+10 
eq_trans = (p-p_eq)
output.PointData.append(eq_trans, 'eq_trans')
"""
    programmableFilter1.RequestInformationScript = ''
    programmableFilter1.RequestUpdateExtentScript = ''
    programmableFilter1.PythonPath = ''


def plot_slice_center_viscosity(source_name, snapshot, pv_output_dir, _time, **kwargs):

    # addtional_options
    has_metastable_region = kwargs.get("has_metastable_region", False)
    has_metastable_region_slab = kwargs.get("has_metastable_region_slab", False)

    # Get the active view 
    renderView1 = GetActiveViewOrCreate('RenderView')

    # Select the slice center source
    if "GEOMETRY" == "chunk":
        transform1 = FindSource("%s_transform_%05d" % (source_name, snapshot))
    else:
        transform1 = FindSource("%s_%05d" % (source_name, snapshot))
    
    # Add programmable filter for equilibrium phase transition at 410.0
    if "MODEL_TYPE" == "mow":
        add_eq_410_condition(transform1)
        source_eq_trans = FindSource("programmable_eq")
        contourEq = Contour(registrationName='contour_eq_trans', Input=source_eq_trans)
        contourEq.ContourBy = ['POINTS', 'eq_trans']
        contourEq.Isosurfaces = [0.0]
        contourEqDisplay = Show(contourEq, renderView1, 'GeometryRepresentation')
        ColorBy(contourEqDisplay, None)
        contourEqDisplay.LineWidth = 2.0
        contourEqDisplay.Ambient = 1.0
        
        if FOO00 == 0:
            # default: turn off plot
            Hide(contourEq, renderView1)

    # todo_da
    # Add T contour
    # 1 - 725 C, for blockT of metastable region
    # 2 - 900 C, for slab internal in upper mantle
    # 3 and 4, 1100 and 1300 C, for an envelop of slab in the mantle
    if FOO01:
        if FOO01 == 1:
            contourT = 725.0+273.15
        elif FOO01 == 2:
            contourT = 900.0+273.15
        elif FOO01 == 3:
            contourT = 1100.0+273.15
        elif FOO01 == 4:
            contourT = 1300.0+273.15
        else:
            raise NotImplementedError()
        contourT_block = Contour(registrationName='ContourT_block', Input=transform1)
        contourT_block.ContourBy = ['POINTS', 'T']
        contourT_block.Isosurfaces = [contourT]
        contourT_blockDisplay = Show(contourT_block, renderView1, 'GeometryRepresentation')
        ColorBy(contourT_blockDisplay,"T")
        rescale_transfer_function_combined('T', 273.0, 1673.0)
        fieldLUT = GetColorTransferFunction("T")
        fieldLUT.ApplyPreset("Viridis (matplotlib)", True)
        contourT_blockDisplay.LineWidth = 2.0
        contourT_blockDisplay.Ambient = 1.0

    # Show the slab center plot and viscosities
    SetActiveSource(transform1)
    transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')

    set_viscosity_plot(transform1Display, ETA_MIN, ETA_MAX)
    
    # Adjust the position of the point source and show related annotations.
    # In box with 3-d setup, the 2d glyph is not in the slice plane. The solution is rotate it 90 degree.
    # Adjust glyph properties based on the specified parameters.
    sourceV = FindSource("%s_glyph_%05d" % (source_name, snapshot))
    if "GEOMETRY" == "box" and int("DIMENSION") == 3:
        sourceV.GlyphTransform.Rotate = [90.0, 0.0, 0.0]
        sourceVRE = FindSource("%s_glyph_%05d_representative" % (source_name, snapshot))
        sourceVRE.GlyphTransform.Rotate = [90.0, 0.0, 0.0]
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

    # Adjust glyph property
    # for 2d, use 3 times the sample points because I plotted the whole domain
    # In Box geometry, used a different scale_factor because we first need to maintain the scale of length
    if int("DIMENSION") == 3:
        n_sample_points = 1000
    else:
        n_sample_points = 3000
    if "GEOMETRY" == "chunk":
        point_source_center = [0, 6.4e6, 0]
        scale_factor = 2.8e6
    elif "GEOMETRY" == "box":
        if int("DIMENSION") == 3:
            point_source_center = [4.65e6, 0, 2.95e6 + TOP - 2890e3]
        else:
            point_source_center = [4.65e6, 2.95e6 + TOP - 2890e3, 0]
        scale_factor = 3.21e6
    else:
        raise NotImplementedError()
    adjust_glyph_properties("%s_glyph_%05d" % (source_name, snapshot), scale_factor, n_sample_points, point_source_center)


    # Show the original trench position
    # sourceTrOrigTrian = FindSource("trench_orig_triangle")
    # sourceTrOrigTrianDisplay = Show(sourceTrOrigTrian, renderView1, 'GeometryRepresentation')
    
    # Show the current trench position
    sourceTrTrian = FindSource("trench_triangle")
    sourceTrTrianDisplay = Show(sourceTrTrian, renderView1, 'GeometryRepresentation')

    # Show the metastable area
    # Sub-options for showing total area and cold area in the slab
    if "MODEL_TYPE" == "mow" and has_metastable_region and FOO02 == 1: 
        # Select the metastable_region source
        if "GEOMETRY" == "chunk":
            metaRegion = FindSource("metastable_region_transform_%05d" % (snapshot))
        else:
            metaRegion = FindSource("metastable_region_%05d" % (snapshot))
        metaRegionDisplay = Show(metaRegion, renderView1, 'GeometryRepresentation')
        ColorBy(metaRegionDisplay, None)
        metaRegionDisplay.AmbientColor = [0.5, 0.5, 0.5] # gray
        metaRegionDisplay.DiffuseColor = [0.5, 0.5, 0.5] # gray
    
    if "MODEL_TYPE" == "mow" and has_metastable_region_slab and FOO03 == 1: 
        # Select the metastable_region source
        if "GEOMETRY" == "chunk":
            metaRegion1 = FindSource("metastable_region_slab_transform_%05d" % (snapshot))
        else:
            metaRegion1 = FindSource("metastable_region_slab_%05d" % (snapshot))
        metaRegion1Display = Show(metaRegion1, renderView1, 'GeometryRepresentation')
        ColorBy(metaRegion1Display, None)
        metaRegion1Display.AmbientColor = [1.0, 0.0, 1.0] # magenta
        metaRegion1Display.DiffuseColor = [1.0, 0.0, 1.0]
        

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
        # todo_center
        if int("DIMENSION") == 3:
            renderView1.CameraPosition = [4012781.8124555387 + (SP_AGE - 80e6)*0.05, 8150298.613651189, 2374146.9531437973 + TOP - 2890e3]
            renderView1.CameraFocalPoint = [4012781.8124555387 + (SP_AGE - 80e6)*0.05, 0.0, 2374146.9531437973 + TOP - 2890e3]
            renderView1.CameraViewUp = [2.220446049250313e-16, 0.0, 1.0]
        else:
            renderView1.CameraPosition = [4012781.8124555387 + (SP_AGE - 80e6)*0.05, 2374146.9531437973 + TOP - 2890e3, -8150298.613651189]
            renderView1.CameraFocalPoint = [4012781.8124555387 + (SP_AGE - 80e6)*0.05, 2374146.9531437973 + TOP - 2890e3, 0.0]
            renderView1.CameraViewUp = [2.220446049250313e-16, 1.0, 0.0]
        renderView1.CameraParallelScale = 802823.959456

    # save figure
    fig_path = os.path.join(pv_output_dir, "slice_center_viscosity_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "slice_center_viscosity_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)
    
    # Plot the density
    # get opacity transfer function/opacity map for 'field'
    Hide(sourceV, renderView1)
    # Hide(sourceTrOrigTrian, renderView1)
    Hide(sourceTrTrian, renderView1)
    fieldLUT = GetColorTransferFunction("viscosity")
    fieldPWF = GetOpacityTransferFunction("viscosity")
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)
    HideScalarBarIfNotNeeded(fieldPWF, renderView1)
    set_density_plot(transform1Display)
    # save figure
    fig_path = os.path.join(pv_output_dir, "slice_center_density_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "slice_center_density_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # Plot the temperature
    # get opacity transfer function/opacity map for 'field'
    Hide(sourceV, renderView1)
    # Hide(sourceTrOrigTrian, renderView1)
    Hide(sourceTrTrian, renderView1)
    fieldLUT = GetColorTransferFunction("density")
    fieldPWF = GetOpacityTransferFunction("density")
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)
    HideScalarBarIfNotNeeded(fieldPWF, renderView1)
    set_temperature_plot(transform1Display)
    # save figure
    fig_path = os.path.join(pv_output_dir, "slice_center_temperature_t%.4e.pdf" % _time)
    fig_png_path = os.path.join(pv_output_dir, "slice_center_temperature_t%.4e.png" % _time)
    SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
    ExportView(fig_path, view=renderView1)
    print("Figure saved: %s" % fig_png_path)
    print("Figure saved: %s" % fig_path)

    # Plot the dynamic pressure
    Hide(sourceV, renderView1)
    # Hide(sourceTrOrigTrian, renderView1)
    Hide(sourceTrTrian, renderView1)
    fieldLUT = GetColorTransferFunction("temperature")
    fieldPWF = GetOpacityTransferFunction("temperature")
    HideScalarBarIfNotNeeded(fieldLUT, renderView1)
    HideScalarBarIfNotNeeded(fieldPWF, renderView1)
    # save figure for slab
    if HAS_DYNAMIC_PRESSURE:
        set_non_adiabatic_pressure_plot_slab(transform1Display)
        fig_path = os.path.join(pv_output_dir, "slice_center_nP_slab_t%.4e.pdf" % _time)
        fig_png_path = os.path.join(pv_output_dir, "slice_center_nP_slab_t%.4e.png" % _time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)
        print("Figure saved: %s" % fig_png_path)
        print("Figure saved: %s" % fig_path)
        # save figure for mantle
        set_non_adiabatic_pressure_plot_mantle(transform1Display)
        fig_path = os.path.join(pv_output_dir, "slice_center_nP_mantle_t%.4e.pdf" % _time)
        fig_png_path = os.path.join(pv_output_dir, "slice_center_nP_mantle_t%.4e.png" % _time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)
        print("Figure saved: %s" % fig_png_path)
        print("Figure saved: %s" % fig_path)
        # reset colorbar
        fieldLUT = GetColorTransferFunction("nonadiabatic_pressure")
        fieldPWF = GetOpacityTransferFunction("nonadiabatic_pressure")
        HideScalarBarIfNotNeeded(fieldLUT, renderView1)
        HideScalarBarIfNotNeeded(fieldPWF, renderView1)

    # Plot the mow contents. This options is only affective when there is
    # "metastable" presented in the name of compositions.
    if "MODEL_TYPE" == "mow":
        # hide old color bars
        # HideAllScalarBars(renderView1)

        # set up metastable plot
        set_metastable_plot(transform1Display)
        # save figure
        fig_path = os.path.join(pv_output_dir, "slice_center_mow_t%.4e.pdf" % _time)
        fig_png_path = os.path.join(pv_output_dir, "slice_center_mow_t%.4e.png" % _time)
        SaveScreenshot(fig_png_path, renderView1, ImageResolution=layout_resolution)
        ExportView(fig_path, view=renderView1)
        print("Figure saved: %s" % fig_png_path)
        print("Figure saved: %s" % fig_path)

    # hide objects
    if ANIMATION:
        hide_everything()


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
    # sourceTrOrigTrian = FindSource("trench_orig_triangle")
    # sourceTrOrigTrianDisplay = Show(sourceTrOrigTrian, renderView1, 'GeometryRepresentation')
    
    # Show the current trench position
    sourceTrTrian = FindSource("trench_triangle")
    sourceTrTrianDisplay = Show(sourceTrTrian, renderView1, 'GeometryRepresentation')

    # Show the trench position
    # sourceTr = FindSource("trench_d0.00km%s_%05d" % (trailer, snapshot))
    # sourceTrDisplay = Show(sourceTr, renderView1, 'GeometryRepresentation')
    # sourceTrDisplay.AmbientColor = [0.3333333333333333, 0.0, 0.0] # Dark red
    # sourceTrDisplay.DiffuseColor = [0.3333333333333333, 0.0, 0.0]

    # Show the trench position
    sourceTr1 = FindSource("trench_d50.00km%s_%05d" % (trailer, snapshot))
    sourceTr1Display = Show(sourceTr1, renderView1, 'GeometryRepresentation')
    sourceTr1Display.AmbientColor = [0.3333333333333333, 0.0, 0.0] # Dard red # [1.0, 0.75, 0.8]   # Pink
    sourceTr1Display.DiffuseColor = [0.3333333333333333, 0.0, 0.0] # Dark red  # Pink

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
        y_diff = np.max([SLAB_EXTENTS_FULL - 1000e3, 0.0])
        renderView1.CameraPosition = [7901107.27312585, 5768057.049286575 + y_diff, 5496301.138370097 + OUTER_RADIUS-2890e3]
        renderView1.CameraFocalPoint = [2242318.5206255154, -1901461.3695231928 + y_diff, 369527.98904717574 + OUTER_RADIUS-2890e3]
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

    
    # hide objects
    if ANIMATION:
        hide_everything()

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
        load_pyvista_source(data_output_dir, "slice_depth_200.0km", snapshot, file_type="vtu", assign_field=True, add_glyph=True)

        # load subducting plate 
        load_pyvista_source(data_output_dir, "sp_lower_above_0.8_filtered_pe", snapshot, file_type="vtu", assign_field=True)
        
        # load slab surfaces
        load_pyvista_source(data_output_dir, "sp_upper_surface", snapshot, file_type="vtp", assign_field=True)
        load_pyvista_source(data_output_dir, "sp_lower_surface", snapshot, file_type="vtp", assign_field=True)
        
        # load trench position
        load_pyvista_source(data_output_dir, "trench_d0.00km", snapshot, file_type="vtp")
        load_pyvista_source(data_output_dir, "trench_d50.00km", snapshot, file_type="vtp")


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

        XMLPartitionedUnstructuredGridReader(registrationName='solution_%05d' % snapshot, FileName=[filein])
        
        # add rotation
        if "GEOMETRY" == "chunk":
            registration_name_transform = 'solution_transform_%05d' % (snapshot)
            solution = FindSource("solution")
            transform = Transform(registrationName=registration_name_transform, Input=solution)
            transform.Transform = 'Transform'
            transform.Transform.Translate = [0.0, 0.0, 0.0]  # center of rotation
            transform.Transform.Rotate = [0.0, 0.0, ROTATION_ANGLE]  # angle of rotation
            Hide3DWidgets()

        registration_name_glyph = 'solution_glyph_%05d' % (snapshot)
        if "GEOMETRY" == "chunk":
            add_glyph1("solution_transform_%05d" % (snapshot), "velocity", 1e6, registrationName=registration_name_glyph)
        else:
            add_glyph1("solution_%05d" % (snapshot), "velocity", 1e6, registrationName=registration_name_glyph)

        # add source of metastable region
        # 1. total metastable area
        # 2. cold metastable area in the slab
        # Note these options will generate separate flags for the plotting function
        has_metastable_region = False
        has_metastable_region_slab = False
        if "MODEL_TYPE" == "mow":
            filein = os.path.join(data_output_dir, "..", "pyvista_outputs", "%05d"%snapshot, "metastable_region_%05d.vtu" %snapshot) 
            if os.path.isfile(filein):
                XMLUnstructuredGridReader(registrationName='metastable_region_%05d' % snapshot, FileName=[filein])
                if "GEOMETRY" == "chunk":
                    registration_name_transform = 'metastable_region_transform_%05d' % (snapshot)
                    solution = FindSource('metastable_region_%05d' % snapshot)
                    transform = Transform(registrationName=registration_name_transform, Input=solution)
                    transform.Transform = 'Transform'
                    transform.Transform.Translate = [0.0, 0.0, 0.0]  # center of rotation
                    transform.Transform.Rotate = [0.0, 0.0, ROTATION_ANGLE]  # angle of rotation
                    Hide3DWidgets()
                else:
                    pass
                has_metastable_region = True
            
            filein = os.path.join(data_output_dir, "..", "pyvista_outputs", "%05d"%snapshot, "metastable_region_slab_%05d.vtu" %snapshot) 
            if os.path.isfile(filein):
                XMLUnstructuredGridReader(registrationName='metastable_region_slab_%05d' % snapshot, FileName=[filein])
                if "GEOMETRY" == "chunk":
                    registration_name_transform = 'metastable_region_slab_transform_%05d' % (snapshot)
                    solution = FindSource('metastable_region_slab_%05d' % snapshot)
                    transform = Transform(registrationName=registration_name_transform, Input=solution)
                    transform.Transform = 'Transform'
                    transform.Transform.Translate = [0.0, 0.0, 0.0]  # center of rotation
                    transform.Transform.Rotate = [0.0, 0.0, ROTATION_ANGLE]  # angle of rotation
                    Hide3DWidgets()
                else:
                    pass
                has_metastable_region_slab = True

        # plot slice
        plot_slice_center_viscosity("solution", snapshot, pv_output_dir, _time, has_metastable_region=has_metastable_region,\
                                    has_metastable_region_slab=has_metastable_region_slab)


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