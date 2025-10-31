def twod_workflow(pv_output_dir, data_output_dir, steps, times):
    '''
    Workflow for the twod case
    Inputs:
        pv_output_dir - directory for figure
        data_output_dir - where the original case output locates
        steps - time steps
        times - corresponding times
    '''
    for i, step in enumerate(steps):
        if PP_INITIAL_REFINEMENT:
            snapshot = INITIAL_ADAPTIVE_REFINEMENT+step
        else:
            snapshot = step
        _time = times[i]
    
        # add source
        filein = os.path.join(data_output_dir, "solution", "solution-%05d.pvtu" %snapshot) 

        XMLPartitionedUnstructuredGridReader(registrationName='solution_%05d' % snapshot, FileName=[filein])
        
    pass

def thd_workflow(pv_output_dir, data_output_dir, steps, times):
    '''
    Workflow for the thd case
    Inputs:
        pv_output_dir - directory for figure
        data_output_dir - where the original case output locates
        steps - time steps
        times - corresponding times
    '''
    # placeholder
    pass


steps = GRAPHICAL_STEPS
times = GRAPHICAL_TIMES
data_output_dir = "OUTPUT_DIRECTORY"
pv_output_dir = os.path.abspath(os.path.join("IMAGE_DIRECTORY", "pv_outputs"))

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
    

if int("DIMENSION") == 3:
    thd_workflow(pv_output_dir, data_output_dir, steps, times)
else:
    twod_workflow(pv_output_dir, data_output_dir, steps, times)