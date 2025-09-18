import pytest
import filecmp  # for compare file contents
from shutil import rmtree
from hamageolib.research.haoyuan_3d_subduction.post_process import *
from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS_TWOD1
import psutil, os

test_dir = os.path.join(".test", "haoyuan_3d_subduction_post_process")
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)

@pytest.mark.big_test  # Optional marker for big tests
@pytest.mark.parametrize(
    "vq0, vq1, expected",
    [
        (6371e3, 1.0*np.pi/180.0, 6.545300e-01),
        (6321e3, 1.0*np.pi/180.0, 6.642250e-01),
        (6371e3, 15.0*np.pi/180.0, 6.545300e-01),
        (6321e3, 15.0*np.pi/180.0, 6.739218e-01),
        (6371e3, 30.0*np.pi/180.0, np.nan)
    ]
)
def test_interpolate_slab_surface(vq0, vq1, expected):
    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_bw8000_sw2000_yd500.0_AR4")
    slab_surface_file = os.path.join(local_dir, "sp_upper_surface_test_00002.vtp")

    scale0 = 1000e3; scale1 = 1000e3/6371e3

    slab_surface = pv.read(slab_surface_file)
    points = slab_surface.points

    v0, v2, v1 = PUnified.points2unified3(points, True, False)
    slab_surface_interp_func = KNNInterpolatorND(np.vstack((v0/scale0, v1/scale1)).T, v2, k=1, max_distance=0.05)

    vq2 = slab_surface_interp_func(vq0/scale0, vq1/scale1)
    if ~np.isnan(expected):
        assert pytest.approx(vq2, rel=1e-5) == expected, f"Failed for ({vq0},{vq1}), expected {expected}, computed {vq2}"
    else:
        assert np.isnan(vq2), f"Failed for ({vq0},{vq1}), expected {expected}, computed {vq2}"

@pytest.mark.parametrize(
    "slab_surface_file_base, depth, std_file",
    [
        ("sp_upper_surface_test_00002.vtp", 0.0, "trench_d0.00km_00002_std.vtp"),
        ("sp_upper_surface_test_00002.vtp", 50e3, "trench_d50.00km_00002_std.vtp"),
        ("sp_upper_surface_test_00014.vtp", 0.0, "trench_d0.00km_00014_std.vtp"),
        ("sp_upper_surface_test_00014.vtp", 50e3, "trench_d50.00km_00014_std.vtp")
    ]
)
@pytest.mark.big_test  # Optional marker for big tests
def test_interpolate_trench_profile(slab_surface_file_base, depth, std_file):
    local_dir=os.path.join("big_tests", "ThDSubduction", "test_interpolate_trench_profile")
    slab_surface_file = os.path.join(local_dir, slab_surface_file_base)
    slab_surface_file_basename = slab_surface_file_base.split(".")[0]
    
    # remove old directory 
    pyvista_outdir = os.path.join(test_dir, "test_interpolate_trench_profile")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)
    os.mkdir(pyvista_outdir)

    scale0 = 1000e3; scale1 = 1000e3/6371e3

    slab_surface = pv.read(slab_surface_file)
    points = slab_surface.points

    v0, v2, v1 = PUnified.points2unified3(points, True, False)
    slab_surface_interp_func = KNNInterpolatorND(np.vstack((v0/scale0, v1/scale1)).T, v2, k=1, max_distance=0.05)

    # K-nearneighbor interpolation
    # test 1 - test interpolation at the surface
    N1 = 1000
    Min1 = 0.0; Max1 = 71.94572847349845*np.pi/180.0
    val1s = np.linspace(Min1, Max1, N1)
    val0s = np.full(val1s.shape, 6371e3 - depth)
    val2s = slab_surface_interp_func(val0s/scale0, val1s/scale1)

    mask = ~np.isnan(val2s)
    x, y, z = PUnified.unified2points3(np.vstack((val0s[mask], val2s[mask], val1s[mask])).T, True, False)
    trench_points = np.vstack([x, y, z]).T

    point_cloud_tr = pv.PolyData(trench_points)
    filename = "%s_trench_d%.2fkm.vtp" % (slab_surface_file_basename, depth/1e3)
    filepath = os.path.join(pyvista_outdir, filename)
    point_cloud_tr.save(filepath)

    assert(os.path.isfile(filepath))
    assert(filecmp.cmp(filepath, os.path.join(local_dir, std_file)))


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_chunk():

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_bw8000_sw2000_yd500.0_AR4")
    pvtu_step=2
    
    # remove old directory 
    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_box")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)

    # initiate the object
    config = {"geometry": "chunk", "Max0": 6371e3, "Min0": 3.4810e+06,\
         "Max1": 71.94572847349845*np.pi/180.0, "Max2": 80.00365006253027*np.pi/180.0, "time":0.0}
    kwargs = {"pyvista_outdir": os.path.join(test_dir, "test_pyvista_process_thd_chunk")}
    PprocessThD = PYVISTA_PROCESS_THD(os.path.join(local_dir, "output", "solution"), config, **kwargs)
    # read vtu file
    PprocessThD.read(pvtu_step)
    # slice at center
    PprocessThD.slice_center()
    # slice at surface
    PprocessThD.slice_surface()
    # slice at depth
    PprocessThD.slice_at_depth(depth=200e3, r_diff=20e3)
    # extract sp_upper composition beyond a threshold
    PprocessThD.extract_iso_volume_upper(threshold=0.8)
    # extract sp_lower composition beyond a threshold
    PprocessThD.extract_iso_volume_lower(threshold=0.8)
    PprocessThD.get_slab_depth()
    # extract plate_edge composition beyond a threshold
    PprocessThD.extract_plate_edge(threshold=0.8)
    # extract slab surface

    # extract slab surface 
    PprocessThD.extract_slab_interface("sp_upper", dr=0.001)
    # extract slab edge
    PprocessThD.extract_plate_edge_surface()
    # filter the slab lower points
    PprocessThD.filter_slab_lower_points()
    # Slab analysis
    PprocessThD.extract_slab_dip_angle_deprecated_0()
    PprocessThD.extract_slab_trench()

    # assert slab dip and trench location
    # with this resolution, we can get the surface trench point, but the 50 depth point
    # is not found
    trench_center0 = 0.6545283794403076
    assert(abs((PprocessThD.trench_center-trench_center0)/trench_center0) < 1e-6)
    slab_depth0 = 240834.0
    assert(abs((PprocessThD.slab_depth-slab_depth0)/slab_depth0) < 1e-6)
    dip_100_center0 = 0.5782019954172547
    assert(abs((PprocessThD.dip_100_center-dip_100_center0)/dip_100_center0) < 1e-6)

    # compare file outputs
    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_chunk")
    depth_slice_file_std = os.path.join(local_dir, "slice_depth_200.0km_00002_std.vtu")
    depth_slice_file = os.path.join(pyvista_outdir, "slice_depth_200.0km_00002.vtu")
    assert(os.path.isfile(depth_slice_file_std))
    assert(filecmp.cmp(depth_slice_file, depth_slice_file_std))
    
    sp_upper_file_std = os.path.join(local_dir, "sp_upper_surface_00002_std.vtp")
    sp_upper_file = os.path.join(pyvista_outdir, "sp_upper_surface_00002.vtp")
    assert(os.path.isfile(sp_upper_file))
    assert(filecmp.cmp(sp_upper_file, sp_upper_file_std))
    
    sp_plate_file_std = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00002_std.vtu")
    sp_plate_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_00002.vtu")
    assert(os.path.isfile(sp_plate_file))
    assert(filecmp.cmp(sp_plate_file, sp_plate_file_std))
    
    trench_file_std = os.path.join(local_dir, "trench_d0.00km_00002_std.vtp")
    trench_file = os.path.join(pyvista_outdir, "trench_d0.00km_00002.vtp")
    assert(os.path.isfile(trench_file))
    assert(filecmp.cmp(trench_file, trench_file_std))

@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_chunk_dip_angle():

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_bw8000_sw2000_yd500.0_AR4")
    pvtu_step=2
    
    # remove old directory 
    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_box")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)

    # initiate the object
    config = {"geometry": "chunk", "Max0": 6371e3, "Min0": 3.4810e+06,\
         "Max1": 71.94572847349845*np.pi/180.0, "Max2": 80.00365006253027*np.pi/180.0, "time":0.0}
    kwargs = {"pyvista_outdir": os.path.join(test_dir, "test_pyvista_process_thd_chunk")}
    PprocessThD = PYVISTA_PROCESS_THD(os.path.join(local_dir, "output", "solution"), config, **kwargs)
    # read vtu file
    PprocessThD.read(pvtu_step)
    # slice at center
    PprocessThD.slice_center()
    # slice at surface
    PprocessThD.slice_surface()
    # slice at depth
    PprocessThD.slice_at_depth(depth=200e3, r_diff=20e3)
    # extract sp_upper composition beyond a threshold
    PprocessThD.extract_iso_volume_upper(threshold=0.8)
    # extract sp_lower composition beyond a threshold
    PprocessThD.extract_iso_volume_lower(threshold=0.8)
    PprocessThD.get_slab_depth()
    # extract plate_edge composition beyond a threshold
    PprocessThD.extract_plate_edge(threshold=0.8)
    # extract slab surface

    # extract slab surface 
    PprocessThD.extract_slab_interface("sp_upper", dr=0.001)
    # extract slab edge
    PprocessThD.extract_plate_edge_surface()
    # filter the slab lower points
    PprocessThD.filter_slab_lower_points()
    # extract slab moho
    PprocessThD.extract_slab_interface("sp_lower", dr=0.001)
    # Slab analysis
    PprocessThD.extract_slab_dip_angle()

    # check value of slab dip angle
    dip_100_center0 = 0.9466176634879051
    assert(abs((PprocessThD.dip_100_center-dip_100_center0)/dip_100_center0) < 1e-6)

@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_box():

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_c22_AR4")
    pvtu_step=2

    # remove old directory 
    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_box")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)

    # initiate the object
    config = {"geometry": "box", "Max0": 2890000.0, "Min0": 0.0, "Max1": 4000000.0, "Max2": 8896000.0, "time": 0.0}
    kwargs = {"pyvista_outdir": os.path.join(test_dir, "test_pyvista_process_thd_box")}
    PprocessThD = PYVISTA_PROCESS_THD(os.path.join(local_dir, "output", "solution"), config, **kwargs)
    # read vtu file
    pvtu_filepath = os.path.join(local_dir, "output", "solution", "solution-%05d.pvtu" % pvtu_step)
    PprocessThD.read(pvtu_step)
    # slice at center
    PprocessThD.slice_center()
    # slice at surface
    PprocessThD.slice_surface()
    # slice at depth
    PprocessThD.slice_at_depth(depth=200e3, r_diff=20e3)
    # extract sp_upper composition beyond a threshold
    PprocessThD.extract_iso_volume_upper(threshold=0.8)
    # extract sp_lower composition beyond a threshold
    PprocessThD.extract_iso_volume_lower(threshold=0.8)
    PprocessThD.get_slab_depth()
    # extract plate_edge composition beyond a threshold
    PprocessThD.extract_plate_edge(threshold=0.8)
    # extract slab surface
    PprocessThD.extract_slab_interface("sp_upper")
    # extract slab edge
    PprocessThD.extract_plate_edge_surface()
    # filter the slab lower points
    PprocessThD.filter_slab_lower_points()

    # Slab analysis
    PprocessThD.extract_slab_dip_angle_deprecated_0()
    PprocessThD.extract_slab_trench()

    # assert slab dip and trench location
    trench_center0 = 4170000.0
    assert(abs((PprocessThD.trench_center-trench_center0)/trench_center0) < 1e-6)
    slab_depth0 = 240833.25
    assert(abs((PprocessThD.slab_depth-slab_depth0)/slab_depth0) < 1e-6)

    # compare file outputs
    depth_slice_file_std = os.path.join(local_dir, "slice_depth_200.0km_00002_std.vtu")
    depth_slice_file = os.path.join(pyvista_outdir, "slice_depth_200.0km_00002.vtu")
    assert(os.path.isfile(depth_slice_file_std))
    assert(filecmp.cmp(depth_slice_file, depth_slice_file_std))
    
    sp_upper_file_std = os.path.join(local_dir, "sp_upper_surface_00002_std.vtp")
    sp_upper_file = os.path.join(pyvista_outdir, "sp_upper_surface_00002.vtp")
    assert(os.path.isfile(sp_upper_file))
    assert(filecmp.cmp(sp_upper_file, sp_upper_file_std))
    
    sp_plate_file_std = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00002_std.vtu")
    sp_plate_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_00002.vtu")
    assert(os.path.isfile(sp_plate_file))
    assert(filecmp.cmp(sp_plate_file, sp_plate_file_std))

    trench_file_std = os.path.join(local_dir, "trench_d0.00km_00002_std.vtp")
    trench_file = os.path.join(pyvista_outdir, "trench_d0.00km_00002.vtp")
    assert(os.path.isfile(trench_file))
    assert(filecmp.cmp(trench_file, trench_file_std))

@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_box_sp_velocity():

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_1_SA50.0_OA20.0_width80_bw2000_sw500")
    pvtu_step=4

    # remove old directory 
    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_box_sp_velocity")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)

    # initiate the object
    config = {"geometry": "box", "Max0": 2890000.0, "Min0": 0.0, "Max1": 4000000.0, "Max2": 8896000.0, "time": 0.0}
    kwargs = {"pyvista_outdir": os.path.join(test_dir, "test_pyvista_process_thd_box")}
    PprocessThD = PYVISTA_PROCESS_THD(os.path.join(local_dir, "output", "solution"), config, **kwargs)
    # read vtu file
    pvtu_filepath = os.path.join(local_dir, "output", "solution", "solution-%05d.pvtu" % pvtu_step)
    PprocessThD.read(pvtu_step)
    # extract sp_upper composition beyond a threshold
    PprocessThD.extract_iso_volume_upper(threshold=0.8)
    # extract sp_lower composition beyond a threshold
    PprocessThD.extract_iso_volume_lower(threshold=0.8)
    # extract plate_edge composition beyond a threshold
    PprocessThD.extract_plate_edge(threshold=0.8)
    # extract slab surface
    PprocessThD.extract_slab_interface("sp_upper")
    # extract slab edge
    PprocessThD.extract_plate_edge_surface()
    # filter the slab lower points
    PprocessThD.filter_slab_lower_points()
    # extract slab moho
    PprocessThD.extract_slab_interface("sp_lower")
    # Slab analysis
    PprocessThD.extract_slab_trench()
    # Plate velocity
    PprocessThD.extract_sp_velocity()


    assert(math.isclose(PprocessThD.sp_velocity,0.022273129, rel_tol=1e-6))


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_box_big():

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_bw8000_sw2000_c22_AR4")
    pvtu_step=58

    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_box_big")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)

    # initiate the object
    config = {"geometry": "box", "Max0": 2890000.0, "Min0": 0.0, "Max1": 8000000.0, "Max2": 8896000.0, "time": 0.0}
    n_pieces = 16 # debug
    kwargs = {"pyvista_outdir": os.path.join(test_dir, "test_pyvista_process_thd_box_big"), "n_pieces": n_pieces}
    PprocessThD = PYVISTA_PROCESS_THD(os.path.join(local_dir, "output", "solution"), config, **kwargs)
    
    # Apply the piecewise options
    slice_depth = 200e3; r_diff=20e3
    iso_volume_threshold = 0.8

    # first process the slice outputs
    # process sp_upper  and sp_edge
    # extract slab edge surface at the side of the plate
    # process sp_lower    
    # filter the slab lower points based on sp_edge surface
    for piece in range(n_pieces):
        PprocessThD.read(pvtu_step, piece=piece)
        PprocessThD.process_piecewise("slice_center", ["sliced", "sliced_u"])
        PprocessThD.process_piecewise("slice_surface", ["sliced_shell"])
        PprocessThD.process_piecewise("slice_at_depth", ["sliced_depth"], depth=slice_depth, r_diff=r_diff)
        PprocessThD.process_piecewise("extract_iso_volume_upper", ["iso_volume_upper"], threshold=iso_volume_threshold)
        PprocessThD.process_piecewise("extract_plate_edge", ["iso_plate_edge"], threshold=iso_volume_threshold)
        PprocessThD.process_piecewise("extract_iso_volume_lower", ["iso_volume_lower"], threshold=iso_volume_threshold)
  
    PprocessThD.combine_pieces()
 
    PprocessThD.write_key_to_file("sliced", "slice_center_unbounded", "vtp")
    PprocessThD.write_key_to_file("sliced_u", "slice_center", "vtu")
    PprocessThD.write_key_to_file("sliced_shell", "slice_outer", "vtu")
    PprocessThD.write_key_to_file("sliced_depth", "slice_depth_%.1fkm" % (slice_depth/1e3), "vtu")
    PprocessThD.write_key_to_file("iso_volume_upper", "sp_upper_above_%.2f" % (iso_volume_threshold), "vtu")
    PprocessThD.write_key_to_file("iso_plate_edge", "plate_edge_above_%.2f" % (iso_volume_threshold), "vtu")
    PprocessThD.write_key_to_file("iso_volume_lower", "sp_lower_above_%.2f" % (iso_volume_threshold), "vtu")
    
    PprocessThD.extract_plate_edge_surface()
    PprocessThD.extract_slab_interface("sp_upper")
    PprocessThD.filter_slab_lower_points()

    # analysis 
    PprocessThD.extract_slab_dip_angle_deprecated_0()
    PprocessThD.extract_slab_trench()
    PprocessThD.get_slab_depth()

    # check file outputs
    filename = "slice_center_unbounded_%05d.vtp" % pvtu_step
    filepath = os.path.join(pyvista_outdir, filename)
    assert(os.path.isfile(filepath))
    
    filename = "slice_outer_%05d.vtu" % pvtu_step
    filepath = os.path.join(pyvista_outdir, filename)
    assert(os.path.isfile(filepath))

    filename = "slice_depth_200.0km_%05d.vtu" % pvtu_step 
    filepath = os.path.join(pyvista_outdir, filename)
    assert(os.path.isfile(filepath))
    
    filename = "sp_upper_above_%.2f_%05d.vtu" % (iso_volume_threshold, pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)
    assert(os.path.isfile(filepath))
    
    filename = "sp_lower_above_%.2f_%05d.vtu" % (iso_volume_threshold, pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)
    assert(os.path.isfile(filepath))
    
    filename = "plate_edge_above_%.2f_%05d.vtu" % (iso_volume_threshold, pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)
    assert(os.path.isfile(filepath))

    # assert slab dip and trench location
    sp_upper_file = os.path.join(pyvista_outdir, "sp_upper_surface_%05d.vtp" % pvtu_step)
    assert(os.path.isfile(sp_upper_file))

    pe_surface_file = os.path.join(pyvista_outdir, "plate_edge_surface_%05d.vtp" % pvtu_step)
    assert(os.path.isfile(pe_surface_file))
    
    sp_plate_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_%05d.vtu" % pvtu_step)
    assert(os.path.isfile(sp_plate_file))

    trench_file = os.path.join(pyvista_outdir, "trench_d0.00km_%05d.vtp" % pvtu_step)
    assert(os.path.isfile(trench_file))
    
    trench_center0 = 4054166.75
    assert(abs((PprocessThD.trench_center-trench_center0)/trench_center0) < 1e-6)
    slab_depth0 = 707448.0
    assert(abs((PprocessThD.slab_depth-slab_depth0)/slab_depth0) < 1e-6)


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_chunk_script_piece_wise():
    '''
    Test for using linux subprocess and run the post-processing piesewise
    '''
    import subprocess

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_bw8000_sw2000_yd500.0_AR4")
    pvtu_step=10
    n_pieces=16
    
    # remove old directory 
    pyvista_outdir = os.path.join(test_dir, "test_pyvista_process_thd_box_chunk_script_piece_wise")
    if os.path.isdir(pyvista_outdir):
        rmtree(pyvista_outdir)
    os.mkdir(pyvista_outdir)

    for i_piece in range(n_pieces):
        print(["python", "hamageolib/research/haoyuan_3d_subduction/scripts/SlabMorphology.py",
                        '-m', "piece-bash", "-d", local_dir, "-d1", pyvista_outdir, "-s", str(pvtu_step), "-n", str(n_pieces), "-i", str(i_piece)])
        completed_process = subprocess.run(["python", "hamageolib/research/haoyuan_3d_subduction/scripts/SlabMorphology.py",
                        '-m', "piece-bash", "-d", local_dir, "-d1", pyvista_outdir, "-s", str(pvtu_step), "-n", str(n_pieces), "-i", str(i_piece)],\
                            capture_output=True, text=True, check=True)

    print(["python", "hamageolib/research/haoyuan_3d_subduction/scripts/SlabMorphology.py",
                    '-m', "piece-bash", "-d", local_dir, "-d1", pyvista_outdir, "-s", str(pvtu_step), "-n", str(n_pieces), "-i", "-1"]) 
    subprocess.run(["python", "hamageolib/research/haoyuan_3d_subduction/scripts/SlabMorphology.py",
                    '-m', "piece-bash", "-d", local_dir, "-d1", pyvista_outdir, "-s", str(pvtu_step), "-n", str(n_pieces), "-i", "-1"], capture_output=True, text=True, check=True)

    # assert output exist 
    odir = os.path.join(pyvista_outdir, "00014")
    assert(os.path.isdir(odir))

    # check slab surface file
    sp_upper_surface_file = os.path.join(odir, "sp_upper_surface_00014.vtp")
    assert(os.path.isfile(sp_upper_surface_file))
    sp_upper_surface_std_file = os.path.join(local_dir, "sp_upper_surface_00014_std.vtp")
    assert(filecmp.cmp(sp_upper_surface_file, sp_upper_surface_std_file))

    # check slab lower file
    sp_lower_above_filtered_pe_file = os.path.join(odir, "sp_lower_above_0.8_filtered_pe_00014.vtu")
    assert(os.path.isfile(sp_lower_above_filtered_pe_file))
    sp_lower_above_filtered_pe_std_file = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00014_std.vtu")
    assert(filecmp.cmp(sp_lower_above_filtered_pe_file, sp_lower_above_filtered_pe_std_file))


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_twod_chunk():
    # test processing the 2d case
    local_dir_2d=os.path.join("big_tests", "ThDSubduction", "eba_cdpt_coh300_SA80.0_OA40.0_width80_ss100.0")
    pvtu_step = 4

    Case_Options_2d = CASE_OPTIONS_TWOD1(local_dir_2d)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)

    # check the output of slab morphology
    dip_100_std = 0.5354595196457564
    assert(abs((output_dict["dip_100"]-dip_100_std)/dip_100_std)<1e-6)

    trench_center_std = 0.6460465192794801
    assert(abs((output_dict["trench_center"]-trench_center_std)/trench_center_std)<1e-6)

    slab_depth_std = 230000.0
    assert(abs((output_dict["slab_depth"]-slab_depth_std)/slab_depth_std)<1e-6)

    sp_velocity_std = 0.03486241
    assert(abs((output_dict["sp_velocity"]-sp_velocity_std)/sp_velocity_std)<1e-6)




@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_twod_box():
    # test processing the 2d case
    local_dir_2d=os.path.join("big_tests", "ThDSubduction", "eba_cdpt_coh300_SA80.0_OA40.0_width80_sc22")
    pvtu_step = 4

    Case_Options_2d = CASE_OPTIONS_TWOD1(local_dir_2d)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)

    # check the output of slab morphology
    dip_100_std = 0.5222396851976059
    assert(abs((output_dict["dip_100"]-dip_100_std)/dip_100_std)<1e-6)

    trench_center_std = 4096639.0
    assert(abs((output_dict["trench_center"]-trench_center_std)/trench_center_std)<1e-6)

    slab_depth_std = 240000.0
    assert(abs((output_dict["slab_depth"]-slab_depth_std)/slab_depth_std)<1e-6)
    
    sp_velocity_std = 0.040054068
    assert(abs((output_dict["sp_velocity"]-sp_velocity_std)/sp_velocity_std)<1e-6)
