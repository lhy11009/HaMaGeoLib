Frozen regression data for the pre-longitude-options generator.

Source implementation: local main at 409e485ffd6c7fabf062a4b33df51bb53ee7675c.
Input case.prm/case.wb were copied from the notebook's current source:
/mnt/lochy/ASPECT_DATA/MOW/test_prescribe/3d_spherical_subduction_along_surface_velocity_2

100000/ and 300000/ contain outputs generated BEFORE changing the generator.
They cover the original _3 benchmark and the notebook's current 300 km setup.
geometry.npz is compressed for storage; arrays are compared numerically.
Only source/destination directory metadata was removed from report.json.
These fixtures do not require /mnt or native WorldBuilder to run the tests.
Do not regenerate them using a changed implementation merely to pass a test.

source/case.prm deliberately retains the original trailing blank line so
the byte-for-byte parameter preservation contract is covered by the tests.
