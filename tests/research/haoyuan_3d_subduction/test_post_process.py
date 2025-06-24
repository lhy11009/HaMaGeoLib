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
    PprocessThD = PYVISTA_PROCESS_THD(pyvista_outdir=pyvista_outdir)
    # read vtu file
    pvtu_filepath = os.path.join(local_dir, "output", "solution", "solution-%05d.pvtu" % pvtu_step)
    PprocessThD.read(pvtu_step, pvtu_filepath)
    # slice at center
    PprocessThD.slice_center()
    # slice at surface
    PprocessThD.slice_surface()
    # slice at depth
    PprocessThD.slice_at_depth(200e3)
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
    slab_surface_file_std = os.path.join(local_dir, "sp_upper_surface_00002_std.vtp")
    slab_surface_file = os.path.join(pyvista_outdir, "sp_upper_surface_00002.vtp")
    assert(os.path.isfile(slab_surface_file))
    assert(filecmp.cmp(slab_surface_file, slab_surface_file_std))

    slab_surface_file_std = os.path.join(local_dir, "sp_lower_above_0.8_filtered_pe_00002_std.vtu")
    slab_surface_file = os.path.join(pyvista_outdir, "sp_lower_above_0.8_filtered_pe_00002.vtu")
    assert(os.path.isfile(slab_surface_file))
    assert(filecmp.cmp(slab_surface_file, slab_surface_file_std))