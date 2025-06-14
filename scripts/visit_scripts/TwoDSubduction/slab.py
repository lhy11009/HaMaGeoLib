# variable to substitute
# vts file:
#   VISIT_FILE
# initial adaptive refinement level:
#   INITIAL_ADAPTIVE_REFINEMENT
# directory for images:
#   IMG_OUTPUT_DIR
# directory for output particle information
#   PARTICLE_OUTPUT_DIR
# all available snapshots for graphical output
#   ALL_AVAILABLE_GRAPHICAL_SNAPSHOTS
# all available snapshots for particle output
#   ALL_AVAILABLE_PARTICLE_SNAPSHOTS
# also ploat the shallower parts below the crust
#   IF_PLOT_SHALLOw
# steps to visualize fields
#   GRAPHICAL_STEPS
# minimum rheology in the domain
#   ETA_MIN
# maximum rheology in the domain
#   ETA_MAX
# rotation of the domain
#   ROTATION_ANGLE
# if peierls rheology is included
#   INCLUDE_PEIERLS_RHEOLOGY
# Geometry
#   GEOMETRY
# Upper mantle view for box geometry
#   GLOBAL_UPPER_MANTLE_VIEW_BOX
# Has dynamic pressure field
#   HAS_DYNAMIC_PRESSURE

geometry = "GEOMETRY"

if geometry == 'chunk':
    global_trench_view = (-200000, 200000, 6.1e+06, 6.372e+06)
    global_upper_mantle_view = (-1.0e+06, 1.0e+06, 5.4e+06, 6.4e+06)
    global_shear_zone_view = (-67000, 136000, 6.2e+06, 6.4e+06) # note this only applies to the first step
    global_full_mantle=(-6.71475e+06, 4.40508e+06, -2.89728e+06, 8.0711e+06)
elif geometry == 'box':
    global_trench_view = (3.98859e+06, 4.21503e+06, 2.77067e+06, 2.99403e+06)  # upper 100 km centered on trench
    global_upper_mantle_view = GLOBAL_UPPER_MANTLE_VIEW_BOX


class SLAB_SPH(VISIT_PLOT):

    def __init__(self, filein, **kwargs):
        """
        initiate
        Args:
            filein(str): path of vtu file
        """
        # get options
        output_dir = kwargs.get('output_dir', '.')

        # types of plot
        plot_types, vars_ = self.get_plots()

        # call __init__ function of parent
        VISIT_PLOT.__init__(self, filein, plot_types=plot_types, vars=vars_, output_dir=output_dir)
        self.eta_min = ETA_MIN
        self.eta_max = ETA_MAX

        # define the field of the slab
        DefineScalarExpression("slab", "spcrust+spharz")
        
        # define a new field if needed
        if INCLUDE_PEIERLS_RHEOLOGY:
            prt1 = "if(lt(peierls_viscosity, dislocation_viscosity), 3.0, 1.0)"
            prt2 = "if(lt(peierls_viscosity, diffusion_viscosity), 3.0, 0.0)"
            prt3 = "2.0"
        else:
            prt1 = "1.0"
            prt2 = "0.0"
            prt3 = "2.0"
        deformation_mechanism_argument = \
        "if(lt(viscosity, 0.99e24), if(lt(dislocation_viscosity,diffusion_viscosity), %s, %s), %s)"\
        % (prt1, prt2, prt3)
        if IF_DEFORM_MECHANISM:
            DefineScalarExpression("deform_mechanism", deformation_mechanism_argument)
            self.add_plot("Pseudocolor", "deform_mechanism")
        
        # set transformation
        if geometry == 'chunk':
            self.set_rotation_transform(ROTATION_ANGLE)

        # set thresholds
        threshold_idxs_tuple=tuple([self.idxs['spcrust'], self.idxs['spharz']])
        SetActivePlots(threshold_idxs_tuple)
        AddOperator("Threshold", 0)
    
        # draw plots
        all_idxs_tuple = tuple([i for i in range(self.all_idxs)])
        SetActivePlots(all_idxs_tuple)
        DrawPlots()
        HideActivePlots()
    
    def get_plots(self):
        """
        get types of plot
        to be reloaded in children
        Returns:
            plot_types(list)
            vars_(list): variables to plot
        """
        plot_types = ["Mesh", "Pseudocolor", "Pseudocolor", "Pseudocolor", "Pseudocolor", "Pseudocolor", "Vector"]
        vars_ = ["mesh", "spcrust", "spharz", "T", "viscosity", "strain_rate","velocity"]
        if HAS_DYNAMIC_PRESSURE==1:
            plot_types.append("Pseudocolor")
            vars_.append("nonadiabatic_pressure")
        return plot_types, vars_

    def plot_time_snap(self):
        """
        plot a single time step
        """
        if IF_PLOT_SHALLOW:
            # spcrust
            self.plot_crust()
            # plot slab viscosity
            self.plot_viscosity_slab()
            # plot deform_mechanism 
            if IF_DEFORM_MECHANISM:
                self.plot_deform_mechanism_slab()
        # plot upper mantle
        # viscosity
        self.plot_viscosity_upper_mantle()
        self.plot_strain_rate_upper_mantle()
        self.plot_temperature_upper_mantle()  # temperature
        # deform mechanism 
        if IF_DEFORM_MECHANISM:
            self.plot_deform_mechanism_upper_mantle()

    
    def plot_crust(self):
        '''
        plot crust properties
        '''
        # set camera
        self.set_view_attrs(global_trench_view)

        # threshold for spcrust
        self.set_threshold('spcrust', [0.8, 1e+37])

        # threshold for spharz
        self.set_threshold('spharz', [0.8, 1e+37])
        
        # set min and max value
        self.set_pseudo_color('spcrust', color_table='Reds', limits=[0.0, 1.0])
        self.set_pseudo_color('spharz', color_table='Blues', limits=[0.0, 1.0])

        # show active plots
        SetActivePlots((self.idxs['mesh'], self.idxs['spcrust'], self.idxs['spharz']))
        HideActivePlots()
        
        # Save Plot
        self.save_window('slab_crust_snap', size=[2048, 2048])
        HideActivePlots()

    def plot_viscosity_slab(self):
        '''
        plot viscosity
        '''
        # set camera
        self.set_view_attrs(global_trench_view)
        
        # set up viscosity
        # change to log scale and invert the color table
        self.set_pseudo_color('viscosity', color_table="SCM_roma", invert_color=False, log=True, limits=[self.eta_min, self.eta_max])
       
        # set up velocity
        SetActivePlots(self.idxs['velocity'])
        VectorAtts = VectorAttributes()
        VectorAtts.glyphLocation = VectorAtts.AdaptsToMeshResolution  # AdaptsToMeshResolution, UniformInSpace
        VectorAtts.colorTableName = "BrBG"
        VectorAtts.nVectors = 4000
        VectorAtts.lineWidth = 0
        VectorAtts.scale = 0.025
        VectorAtts.scaleByMagnitude = 1
        VectorAtts.autoScale = 1
        SetPlotOptions(VectorAtts)
        
        SetActivePlots((self.idxs['viscosity'], self.idxs['velocity']))
        HideActivePlots()

        # save plot 
        self.save_window('slab_viscosity_snap')
        HideActivePlots()
    
    def plot_deform_mechanism_slab(self):
        '''
        plot deform mechanism
        '''
        # set camera
        self.set_view_attrs(global_trench_view)
        
        # set up deform_mechanism
        self.set_pseudo_color('deform_mechanism', color_table='viridis', limits=[0.0, 3.0])
       
        SetActivePlots((self.idxs['deform_mechanism']))
        HideActivePlots()

        # save plot 
        self.save_window('slab_deform_mechanism_snap')
        HideActivePlots()

    def plot_viscosity_upper_mantle(self):
        '''
        plot viscosity
        '''

        # set camera
        self.set_view_attrs(global_upper_mantle_view)
        
        # set up viscosity
        # change to log scale and invert the color table
        self.set_pseudo_color('viscosity', color_table="SCM_roma", invert_color=False, log=True, limits=[self.eta_min, self.eta_max])
       
        # set up velocity
        SetActivePlots(self.idxs['velocity'])
        VectorAtts = VectorAttributes()
        VectorAtts = VectorAttributes( )
        VectorAtts.glyphLocation = VectorAtts.UniformInSpace # AdaptsToMeshResolution, UniformInSpace
        VectorAtts.colorTableName = "BrBG"
        VectorAtts.nVectors = 20000
        VectorAtts.lineWidth = 0
        VectorAtts.scale = 0.05
        VectorAtts.scaleByMagnitude = 1
        VectorAtts.autoScale = 1
        SetPlotOptions(VectorAtts)

        SetActivePlots((self.idxs['viscosity'], self.idxs['velocity']))
        HideActivePlots()
        # save plot 
        self.save_window('um_viscosity_snap')
        HideActivePlots()
    
    def plot_strain_rate_upper_mantle(self):
        '''
        plot the strain rate
        '''

        # set camera
        self.set_view_attrs(global_upper_mantle_view)
        
        # set up viscosity
        # change to log scale and invert the color table
        self.set_pseudo_color('strain_rate', color_table="SCM_batlow", invert_color=False, log=True, limits=[1e-18, 1e-12])
       
        # set up velocity
        SetActivePlots(self.idxs['strain_rate'])
        VectorAtts = VectorAttributes()
        VectorAtts = VectorAttributes( )
        VectorAtts.glyphLocation = VectorAtts.UniformInSpace # AdaptsToMeshResolution, UniformInSpace
        VectorAtts.colorTableName = "BrBG"
        VectorAtts.nVectors = 20000
        VectorAtts.lineWidth = 0
        VectorAtts.scale = 0.05
        VectorAtts.scaleByMagnitude = 1
        VectorAtts.autoScale = 1
        SetPlotOptions(VectorAtts)

        SetActivePlots((self.idxs['strain_rate'], self.idxs['velocity']))
        HideActivePlots()
        # save plot 
        self.save_window('um_strain_rate_snap')
        HideActivePlots()
    
    def plot_deform_mechanism_upper_mantle(self):
        '''
        plot deform mechanism in upper mantle
        '''
        # set camera
        self.set_view_attrs(global_upper_mantle_view)
        
        # set up deform_mechanism
        self.set_pseudo_color('deform_mechanism', color_table='viridis')
       
        SetActivePlots((self.idxs['deform_mechanism']))
        HideActivePlots()

        # save plot 
        self.save_window('um_deform_mechanism_snap')
        HideActivePlots()
    
    def plot_temperature_upper_mantle(self):
        '''
        plot deform mechanism in upper mantle
        '''
        # set camera
        self.set_view_attrs(global_upper_mantle_view)
        
        # set up temperature
        self.set_pseudo_color('T', color_table='SCM_vik', limits=[273.0, 2173.0])
       
        SetActivePlots((self.idxs['T']))
        HideActivePlots()

        # save plot 
        self.save_window('um_temperature')
        HideActivePlots()
    
    def plot_nonadiabatic_pressure_upper_mantle(self):
        '''
        plot deform mechanism in upper mantle
        '''
        # set camera
        self.set_view_attrs(global_upper_mantle_view)
        
        # set up temperature
        self.set_pseudo_color('nonadiabatic_pressure', color_table='SCM_berlin', limits=[-1e8, 1e8])
       
        SetActivePlots((self.idxs['nonadiabatic_pressure']))
        HideActivePlots()

        # save plot 
        self.save_window('um_nonadiabatic_pressure')
        HideActivePlots()
        

class EXPORT_PARTICLE(VISIT_PLOT):
    '''
    export particles from visit file
    generate a file with positions of particles for each step
    '''
    
    def __init__(self, filein, **kwargs):
        """
        initiate
        Args:
            filein(str): path of vtu file
        """
        # get options
        output_dir = kwargs.get('output_dir', '.')

        # types of plot
        plot_types, vars_ = self.get_plots()

        # call __init__ function of parent
        VISIT_PLOT.__init__(self, filein, plot_types=plot_types, vars=vars_, output_dir=output_dir)

        # draw plots
        all_idxs_tuple = tuple([i for i in range(self.all_idxs)])
        SetActivePlots(all_idxs_tuple)
        DrawPlots()
        HideActivePlots()
    
    def get_plots(self):
        """
        get types of plot
        to be reloaded in children
        Returns:
            plot_types(list)
            vars_(list): variables to plot
        """
        plot_types = ["Mesh", "Molecule"]
        vars_ = ["mesh", "id"]
        return plot_types, vars_
    
    def plot_time_snap(self):
        # set active
        SetActivePlots((self.idxs['id']))
        # SetActivePlots((self.idxs['mesh'], self.idxs['id']))
        HideActivePlots()

        # export 
        ExportDBAtts = ExportDBAttributes()
        ExportDBAtts.allTimes = 1
        ExportDBAtts.dirname = "PARTICLE_OUTPUT_DIR"
        ExportDBAtts.filename = "visit_particles"
        ExportDBAtts.timeStateFormat = "_%06d"
        ExportDBAtts.db_type = "XYZ"
        ExportDBAtts.db_type_fullname = "XYZ_1.0"
        ExportDBAtts.variables = ("id")
        ExportDBAtts.writeUsingGroups = 0
        ExportDBAtts.groupSize = 48
        ExportDatabase(ExportDBAtts)
        HideActivePlots()



def main():
    if IF_PLOT_SLAB:
        Slab = SLAB_SPH("%s/../output/solution.visit" % _dir, output_dir="%s/../img" % _dir)
        # First number is the number of initial adaptive refinements
        # Second one is the snapshot to plot
        # here we prefer to use a series of snapshots.
        # If this doesn't work, we will use a single snapshot
        steps = GRAPHICAL_STEPS
        if not steps == []:
            for step in steps:
                # check that snapshot is valid
                snapshots = INITIAL_ADAPTIVE_REFINEMENT+step
                if snapshots in ALL_AVAILABLE_GRAPHICAL_SNAPSHOTS:
                    Slab(INITIAL_ADAPTIVE_REFINEMENT+step)
                else:
                    print "step %s is not valid. There is no output" % step
        else:
            Slab(INITIAL_ADAPTIVE_REFINEMENT+SINGLE_SNAPSHOT)
        # Slab.abort()

    # deprecated: as this part(particle) is not included in the prm file anymore
    # Output particles for slab morphology
    # if IF_EXPORT_SLAB_MORPH:
    #    Export_Particle = EXPORT_PARTICLE("VISIT_PARTICLE_FILE", output_dir="DATA_OUTPUT_DIR")
        # Be default, it outputs all steps. So we only needs to go to a single snapshot
    #    Export_Particle(0)
    #    Export_Particle.abort()

main()