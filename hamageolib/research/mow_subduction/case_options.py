import numpy as np
from ...utils.case_options import ModelConfigManager
from ..haoyuan_3d_subduction.case_options import CASE_OPTIONS_TWOD1


class CASE_OPTIONS_TWOD(CASE_OPTIONS_TWOD1):
   def Interpret(self, **kwargs):
      CASE_OPTIONS_TWOD1.Interpret(self, **kwargs)

      # model type
      # Interpret as "mow" if we find "metastable" in the names of compositional fields
      names_of_compositional_fields_str = self.idict["Compositional fields"]["Names of fields"]
      if "metastable" in names_of_compositional_fields_str:
         self.options["MODEL_TYPE"] = "mow"

      if self.options["MODEL_TYPE"] == "mow":
         default_dict = {
            "Phase transition Clapeyron slope": 2e6,
            "Phase transition depth": 410e3,
            "Phase transition temperature": 1740.0
         }
         metastable_dict = self.idict["Material model"].get("metastable", default_dict)
         self.options["CL_PT_EQ"] = metastable_dict.get("Phase transition Clapeyron slope", 2e6)
         self.options["DEPTH_PT_EQ"] = metastable_dict.get("Phase transition depth", 410e3)
         self.options["P_PT_EQ"] = 1.34829e+10
         self.options["T_PT_EQ"] = metastable_dict.get("Phase transition temperature", 1740.0)

   # todo_mow_summary
   def SummaryCaseVtuStep(self, ifile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_TWOD1.SummaryCaseVtuStep(self, ifile)

        # Add new columns you want to add
        new_columns = ["Mow area"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan
