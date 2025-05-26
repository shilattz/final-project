import os

# הכנסת אישורים
os.environ['CLEARML_API_ACCESS_KEY'] = 'VDTN7A5TTCTI5TGVT5G3003GTXLLQU'
os.environ['CLEARML_API_SECRET_KEY'] = 'g24koMUbC0VHVK5ImD4oOpPumE2kvRxhgXmi0aCF_1z8uoxJtg_80U9SSHy92VxyXYw'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'

# עכשיו אפשר לייבא ClearML
from clearml import Task

# אתחול המשימה
task = Task.init(project_name="Trajectory Fault Detection", task_name="Colab Setup Test")

print("success logging to ClearML")
