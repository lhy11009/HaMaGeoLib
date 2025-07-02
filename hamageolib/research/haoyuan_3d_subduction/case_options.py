# todo_data
import numpy as np
from hamageolib.utils.case_options import CASE_OPTIONS

class CASE_OPTIONS_THD(CASE_OPTIONS):

    def __init__(self, case_dir):
        CASE_OPTIONS.__init__(self, case_dir)
    
    def interpret(self):
        return CASE_OPTIONS.interpret(self)

    def SummaryCase(self):
        """
        Generate Case Summary
        """
        CASE_OPTIONS.SummaryCase(self)
        # add specific columns
        self.summary_df["trench"] = np.nan