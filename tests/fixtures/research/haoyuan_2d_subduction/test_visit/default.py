# vts file:
#   VISIT_FILE
# directory for images:
#   IMG_OUTPUT_DIR
# initial adaptive refinement level:
#   INITIAL_ADAPTIVE_REFINEMENT
# all available snapshots for graphical output
#   ALL_AVAILABLE_GRAPHICAL_SNAPSHOTS
# steps to plot
#   GRAPHICAL_STEPS

class TEMPERATURE_PLOT(VISIT_PLOT):
    # a class for plotting temperature
    def __init__(self, filein, **kwargs):
        plot_types, vars_ = self.get_plots()
        # get options
        output_dir = kwargs.get('output_dir', '.')
        # call __init__ function of parent
        VISIT_PLOT.__init__(self, filein, output_dir=output_dir)
        # draw all the plots 
        self.draw_all()
    
    def get_plots(self):
        """
        get types of plot
        to be reloaded in children
        Returns:
            plot_types(list)
            vars_(list): variables to plot
        """
        plot_types = ["Pseudocolor", "Pseudocolor", "Pseudocolor"]
        vars_ = ["T", "density", "viscosity"]
        return plot_types, vars_

    def plot_time_snap(self):
        # plot option for this class
        self.plot_temperature_base() # temperature

def main():
    Temperature_Plot = TEMPERATURE_PLOT("VISIT_FILE", output_dir="IMG_OUTPUT_DIR")
    steps = GRAPHICAL_STEPS
    if type(steps) == list:
        for step in steps:
            # check that snapshot is valid
            snapshots = INITIAL_ADAPTIVE_REFINEMENT+step
            if snapshots in ALL_AVAILABLE_GRAPHICAL_SNAPSHOTS:
                Temperature_Plot(INITIAL_ADAPTIVE_REFINEMENT+step)
            else:
                print "step %s is not valid. There is no output" % step
    else:
        print "step: " + str(steps) + " is not a list. There is no output"
    # Temperature_Plot(0)

main()