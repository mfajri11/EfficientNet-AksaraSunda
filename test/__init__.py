
import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
module_path  = os.path.join(root_path, 'src')
sys.path.append(module_path)

