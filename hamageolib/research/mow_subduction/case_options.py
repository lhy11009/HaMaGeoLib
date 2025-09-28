import numpy as np
from ..haoyuan_3d_subduction.case_options import CASE_OPTIONS_TWOD1
from ..haoyuan_3d_subduction.case_options import CASE_OPTIONS as CASE_OPTIONS_BASE


class CASE_OPTIONS_TWOD(CASE_OPTIONS_TWOD1):
   def Interpret(self, **kwargs):
      CASE_OPTIONS_TWOD1.Interpret(self, **kwargs)

      # model type
      # Interpret as "mow" if we find "metastable" in the names of compositional fields
      metastabl_options = set_metastable_options(self.idict)
      self.options.update(**metastabl_options)


   def SummaryCaseVtuStep(self, ifile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_TWOD1.SummaryCaseVtuStep(self, ifile)

        # Add new columns you want to add
        # Mow area - metastable area
        # Mow area code - metastable area in cold slab
        new_columns = ["Mow area", "Mow area cold", "Sp velocity"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan

class CASE_OPTIONS(CASE_OPTIONS_BASE):

   def Interpret(self, **kwargs):
      CASE_OPTIONS_BASE.Interpret(self, **kwargs)

      # model type
      # Interpret as "mow" if we find "metastable" in the names of compositional fields
      metastabl_options = set_metastable_options(self.idict)
      self.options.update(**metastabl_options)

   def SummaryCaseVtuStep(self, ifile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_BASE.SummaryCaseVtuStep(self, ifile)

        # Add new columns you want to add
        # Mow area - metastable area
        # Mow area code - metastable area in cold slab
        new_columns = ["Mow area center", "Mow area cold center", "MOW volume", "MOW volume cold", "Sp velocity"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan


def set_metastable_options(idict):

      options = {}

      names_of_compositional_fields_str = idict["Compositional fields"]["Names of fields"]
      if "metastable" in names_of_compositional_fields_str:
         options["MODEL_TYPE"] = "mow"
         default_dict = {
            "Phase transition Clapeyron slope": 2e6,
            "Phase transition depth": 410e3,
            "Phase transition temperature": 1740.0
         }
         metastable_dict = idict["Material model"].get("metastable", default_dict)
         options["CL_PT_EQ"] = metastable_dict.get("Phase transition Clapeyron slope", 2e6)
         options["DEPTH_PT_EQ"] = metastable_dict.get("Phase transition depth", 410e3)
         options["P_PT_EQ"] = 1.34829e+10
         options["T_PT_EQ"] = metastable_dict.get("Phase transition temperature", 1740.0)

      return options