from ...utils.case_options import ModelConfigManager
from ..haoyuan_3d_subduction.case_options import CASE_OPTIONS_TWOD1

# todo_config
class CASE_OPTIONS_TWOD(CASE_OPTIONS_TWOD1):
   def Interpret(self, **kwargs):
      CASE_OPTIONS_TWOD1.Interpret(self, **kwargs)

      # model type
      # Interpret as "mow" if we find "metastable" in the names of compositional fields
      names_of_compositional_fields_str = self.idict["Compositional fields"]["Names of fields"]
      if "metastable" in names_of_compositional_fields_str:
         self.options["MODEL_TYPE"] = "mow"