import json
import os
import filecmp
import shutil
from pathlib import Path
from gdmate.aspect.config_engine import RuleEngine
from gdmate.aspect.builtin_rules import CasePathRule
from gdmate.aspect.io import parse_parameters_to_dict, save_parameters_from_dict
from hamageolib.research.haoyuan_collision0.config import GeometryRule, PostProcessorRule, RemoveFluidRule, RemovePeridotiteRule,\
    RheologyRule, WeakLayerRule, SlabRule, SolverRule, PrescribConditionRule

# Resolve the root of the pakage and set up
# test directory
package_root = Path(__file__).resolve().parents[3]
fixture_root = package_root/"tests/fixtures/research/haoyuan_collision"
test_root = package_root/".test"
test_root.mkdir(exist_ok=True)
test_dir = test_root/"research-haoyuan-case-options"
test_dir.mkdir(exist_ok=True)


rules = [PostProcessorRule(), CasePathRule(), RemoveFluidRule(), RemovePeridotiteRule(),
         SlabRule(), GeometryRule(), RheologyRule(), WeakLayerRule(),
         SolverRule(), PrescribConditionRule()]

def test_default_options():
    """
    Test default configurations.
    Here I check that by taking default values in configurations,
    my files return to the one Fritz send me
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_default_options"
    fixture_dir = fixture_root/"test_default_options"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {}
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))

def test_geometry_options():
    """
    Test configurations of geometry size.
    Here I check that by including geometry options in configurations,
    my files differs from the one Fritz send me in domain size, but would
    yield the same resolution
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_geometry_options"
    fixture_dir = fixture_root/"test_geometry_options"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {"use_my_setup_of_postprocess": True, "domain_depth": 1000e3, "repetition_length": 500e3}
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))

def test_remove_fluid():
    """
    Test configurations of removing fluid compositions
    Here I check that by removing fluid compositions in configurations,
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_remove_fluid"
    fixture_dir = fixture_root/"test_remove_fluid"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {"use_my_setup_of_postprocess": True, "remove_fluid": True, "remove_fluid_compositions": ["porosity", "bound_fluid"], "domain_depth": 1000e3, "repetition_length": 500e3}
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))


def test_use_isofurace():
    """
    Test configurations of removing fluid compositions
    Here I check that by removing fluid compositions in configurations,
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_use_isofurace"
    fixture_dir = fixture_root/"test_use_isofurace"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {"use_my_setup_of_postprocess": True, "remove_fluid": True, "remove_fluid_compositions": ["porosity", "bound_fluid"], "domain_depth": 1000e3, "repetition_length": 500e3, "use_isosurfaces": True}
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))


def test_rheology_setup():
    """
    Test configurations of seting the rheology module in the material model
    Here I check that by removing fluid compositions in configurations,
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_rheology_setup"
    fixture_dir = fixture_root/"test_rheology_setup"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {"use_my_setup_of_postprocess": True, "remove_fluid": True, "remove_fluid_compositions": ["porosity", "bound_fluid"],
              "domain_depth": 1000e3, "repetition_length": 500e3, "use_isosurfaces": True, "use_my_setup_of_rheology": True}
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))


def test_weak_layer_setup():
    """
    Test configurations of weak layer on top of the slab surface
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_weak_layer_setup"
    fixture_dir = fixture_root/"test_weak_layer_setup"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {"use_my_setup_of_postprocess": True,
              "remove_fluid": True,  # remove fluid
              "remove_fluid_compositions": ["porosity", "bound_fluid"], # remove fluid
              "remove_peridotite": True, # remove peridotite
              "domain_depth": 1000e3, "repetition_length": 500e3, "use_isosurfaces": True, # geometry
              "use_my_setup_of_rheology": True, # rheology
              "weak_layer_compositions": ["MORB", "sediment", "gabbro"], # weak layer
              "weak_layer_viscosity": 1e20, # weak layer
              "force_weak_layer_max_refinement": True, # weak layer
              }
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))


def test_slab_layers_setup():
    """
    Test configurations of slab layers
    """
    # set up tests
    # case_dir - directory of the test case. Remove the old ones if existing previously
    # template_prm and template_wb - templates to use for the prm and wb file
    # case_prm and case_wb - the prm and wb files generated
    # case_prm_standard and case_wb_standard - the standand output file to compare with
    case_dir = test_dir/"test_slab_layers_setup"
    fixture_dir = fixture_root/"test_slab_layers_setup"

    if case_dir.is_dir():
        shutil.rmtree(case_dir)
    case_dir.mkdir(exist_ok=False)

    template_prm = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/post_compressible_test.prm"
    template_wb = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/original.wb"

    case_prm = case_dir/"case.prm"
    case_wb = case_dir/"case.wb"

    case_prm_standard = fixture_dir/"case.prm"
    case_wb_standard = fixture_dir/"case.wb"

    # Import files to dict objects
    assert(template_prm.is_file())
    with template_prm.open('r') as fin:
        prm_dict = parse_parameters_to_dict(fin, format_entry=True)
    assert(template_wb.is_file())
    with template_wb.open('r') as fin:
        wb_dict = json.load(fin)

    # Apply the rules
    ruleEngine = RuleEngine(rules)
    config = {"use_my_setup_of_postprocess": True,
              "remove_fluid": True,  # remove fluid
              "remove_fluid_compositions": ["porosity", "bound_fluid"], # remove fluid
              "remove_peridotite": True, # remove peridotite
              "domain_depth": 1000e3, "repetition_length": 500e3, "use_isosurfaces": True, # geometry
              "use_my_setup_of_rheology": True, # rheology
              "weak_layer_compositions": ["MORB", "sediment", "gabbro"], # weak layer
              "weak_layer_viscosity": 1e20, # weak layer
              "force_weak_layer_max_refinement": True, # weak layer
              "slab_layer_compositions": ["sediment", "MORB", "gabbro"], # slab
              "slab_layer_depths": [0.0, 4e3, 7.5e3, 15e3], # slab
              }
    contexts = ruleEngine.apply_all(config, prm_dict, wb_dict)

    # Write case prm file
    with case_prm.open('w') as fout:
        save_parameters_from_dict(fout, prm_dict)
    with case_wb.open('w') as fout:
        json.dump(wb_dict, fout, indent=4)

    # Compare with the standard outputs
    assert(filecmp.cmp(case_prm, case_prm_standard))
    assert(filecmp.cmp(case_wb, case_wb_standard))