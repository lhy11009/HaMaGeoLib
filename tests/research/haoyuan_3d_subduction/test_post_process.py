import pytest
import filecmp  # for compare file contents
from shutil import rmtree
from hamageolib.research.haoyuan_3d_subduction.post_process import *

test_dir = os.path.join(".test", "haoyuan_3d_subduction_post_process")
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)

@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd():

    local_dir=os.path.join("big_tests", "ThDSubduction", "eba3d_width80_bw8000_sw2000_yd500.0_AR4")
    pvtu_step=2
    pyvista_outdir=os.path.join(test_dir, "test_pyvista_process_thd")
    os.mkdir(pyvista_outdir)

    # initiate the object
    config = {"Max0": 6371e3, "Min0": 3.4810e+06, "Max1": 35.972864236749224*np.pi/180.0, "Max2": 80.00365006253027*np.pi/180.0}
    PprocessThD = PYVISTA_PROCESS_THD(config, pyvista_outdir=pyvista_outdir)
    # read vtu file
    pvtu_filepath = os.path.join(local_dir, "output", "solution", "solution-%05d.pvtu" % pvtu_step)
    PprocessThD.read(pvtu_step, pvtu_filepath)
    # slice at center
    PprocessThD.slice_center()
    # slice at surface
    PprocessThD.slice_surface()
    # slice at depth
    PprocessThD.slice_at_depth(200e3, r_diff=20e3)
    # extract sp_upper composition beyond a threshold
    PprocessThD.extract_iso_volume_upper(0.8)
    # extract sp_lower composition beyond a threshold
    PprocessThD.extract_iso_volume_lower(0.8)
    # extract plate_edge composition beyond a threshold
    PprocessThD.extract_plate_edge(0.8)
    # extract slab surface
    PprocessThD.extract_slab_surface()
    # extract slab edge
    PprocessThD.extract_plate_edge_surface()
    # filter the slab lower points
    PprocessThD.filter_slab_lower_points()

    # compare file outputs
    depth_slice_file_std = os.path.join(local_dir, "slice_depth_200.0km_00002_std.vtu")
    depth_slice_file = os.path.join(pyvista_outdir, "slice_depth_200.0km_00002.vtu")
    assert(os.path.isfile(depth_slice_file_std))
    assert(filecmp.cmp(depth_slice_file, depth_slice_file_std))
    
    sp_upper_file_std = os.path.join(local_dir, "sp_upper_surface_00002_std.vtp")
    sp_upper_file = os.path.join(pyvista_outdir, "sp_upper_surface_00002.vtp")
    assert(os.path.isfile(sp_upper_file))
    assert(filecmp.cmp(sp_upper_file, sp_upper_file_std))

    sp_lower_file_std = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00002_std.vtu")
    sp_lower_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_00002.vtu")
    assert(os.path.isfile(sp_lower_file))
    assert(filecmp.cmp(sp_lower_file, sp_lower_file_std))
    
    slab_surface_file_std = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00002_std.vtu")
    slab_surface_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_00002.vtu")
    assert(os.path.isfile(slab_surface_file))
    assert(filecmp.cmp(slab_surface_file, slab_surface_file_std))
    
    pe_surface_file_std = os.path.join(local_dir, "plate_edge_surface_00002_std.vtp")
    pe_surface_file = os.path.join(pyvista_outdir, "plate_edge_surface_00002.vtp")
    assert(os.path.isfile(pe_surface_file))
    assert(filecmp.cmp(pe_surface_file, pe_surface_file_std))
    
    sp_plate_file_std = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00002_std.vtu")
    sp_plate_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_00002.vtu")
    assert(os.path.isfile(sp_plate_file))
    assert(filecmp.cmp(sp_plate_file, sp_plate_file_std))