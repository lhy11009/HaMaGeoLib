import os, sys, argparse
import warnings

HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

# Load package modules
from hamageolib.research.haoyuan_3d_subduction.post_process import get_trench_position_from_file, get_slab_depth_from_file,\
    get_slab_dip_angle_from_file, ProcessVtuFileThDStep
from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS

def main():
    parser = argparse.ArgumentParser(
        description="Run geodynamic processing script with different modes."
    )

    # Method option (required)
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="whole",
        choices=["whole", "piece", "piece-bash"],
        help="Method to run the script."
    )
    
    # Piece total number (only relevant if method is piece or piece-bash)
    parser.add_argument(
        "-n", "--n_pieces",
        type=int,
        help="Total number of the piece to run (only valid with method 'piece' or 'piece-bash')."
    )

    # Piece index (only relevant if method is piece or piece-bash)
    parser.add_argument(
        "-i", "--i_piece",
        type=int,
        help="Index of the piece to run (only valid with method 'piece' or 'piece-bash')."
    )

    # Input directory (must exist)
    parser.add_argument(
        "-d", "--indir",
        type=str,
        required=True,
        help="Path to the input directory (must exist)."
    )

    # Input directory (default is None, and we will take args.indir/pyvista_output as the value)
    parser.add_argument(
        "-d1", "--outdir",
        type=str,
        default=None,
        help="Path to the output directory."
    )

     # Step option (default -1: loop all steps)
    parser.add_argument(
        "-s", "--step",
        type=int,
        default=-1,
        help="Step number to process (default: -1, meaning process all steps)."
    )

    args = parser.parse_args()

    # ----------------------------
    # Argument consistency checks
    # ----------------------------
    if args.method == "whole" and (args.i_piece is not None or args.n_pieces is not None):
        parser.error("The --i_piece option cannot be used with method 'whole'.")

    if args.method in ["piece-bash"] and args.i_piece is None:
        parser.error("The --i_piece option must be provided with method 'piece-bash'.")
    elif args.method != "piece-bash" and args.i_piece is not None:
        parser.error("The --i_piece option should only be provided with method 'piece-bash'.")

    if not os.path.isdir(args.indir):
        parser.error(f"Input directory '{args.indir}' does not exist.")


    # Whether to generate pyvista outputs
    prepare_pyvista = True
    
    # Whether to analyze results
    if args.method in ["whole", "piece"]:
        analyze_results = True
    else:
        if args.i_piece < 0:
            analyze_results = True
        else:
            analyze_results = False

    # Initiate the case option class
    Case_Options = CASE_OPTIONS(args.indir)
    Case_Options.Interpret()
    Case_Options.SummaryCaseVtuStep(os.path.join(args.indir, "summary.csv"))

    graphical_steps_np = Case_Options.summary_df["Vtu step"].to_numpy()
    graphical_steps = [int(step) for step in graphical_steps_np]

    # Specify the step if an option is given
    if args.step >= 0:
        assert(args.step in graphical_steps)
        graphical_steps = [args.step]

    # Output directory
    if args.outdir is None:
        odir = os.path.join(args.indir, "pyvista_outputs")
    else:
        odir = args.outdir

    config = {"threshold_lower": 0.8} # options for processing vtu file
    # Generate paraview script
    for i, step in enumerate(graphical_steps):

        # for case EBA_2d_consistent_8_7/eba3d_width80_bw8000_sw2000_c22_AR4
        # continue what failed before
        # if step <= 58:
        #     continue

        # get trench center
        pvtu_step = step + int(Case_Options.options['INITIAL_ADAPTIVE_REFINEMENT']) 
        pyvista_outdir = os.path.join(odir, "%05d" % pvtu_step)

        # processing pyvista
        try:
            if prepare_pyvista:
                # n_pieces tell the function to proceed piece-wise.
                # i_pice tell the function to only process one piece at a time (otherwise loop over all pieces)
                _, outputs = ProcessVtuFileThDStep(args.indir, pvtu_step, Case_Options, odir=odir, n_pieces=args.n_pieces, i_piece=args.i_piece)
        except FileNotFoundError:
            Case_Options.SummaryCaseVtuStepUpdateValue("File found", step, False)
        else:
            if analyze_results:
                trench_center = get_trench_position_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'])
                slab_depth = get_slab_depth_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'], float(Case_Options.options['OUTER_RADIUS']), "sp_lower")
                dip_100_center = get_slab_dip_angle_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'], float(Case_Options.options['OUTER_RADIUS']), "sp_upper", 0.0, 100e3)
                
                Case_Options.SummaryCaseVtuStepUpdateValue("File found", step, True)
                # update value in sumamry
                Case_Options.SummaryCaseVtuStepUpdateValue("Slab depth", step, slab_depth)
                Case_Options.SummaryCaseVtuStepUpdateValue("Trench (center)", step, trench_center)
                Case_Options.SummaryCaseVtuStepUpdateValue("Dip 100 (center)", step, dip_100_center)
        
        break # debug

    if analyze_results:
        Case_Options.SummaryCaseVtuStepExport(os.path.join(args.indir, "summary.csv"))


if __name__ == "__main__":
    main()