# vts file:
#   VISIT_FILE
# directory for images:
#   IMG_OUTPUT_DIR
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
        plot_types = ["Pseudocolor"]
        vars_ = ["T"]
	return plot_types, vars_

    def plot_time_snap(self):
        # plot option for this class
        self.plot_temperature_base()

def main():
    Temperature_Plot = TEMPERATURE_PLOT("VISIT_FILE", output_dir="IMG_OUTPUT_DIR")
    Temperature_Plot(0)

main()