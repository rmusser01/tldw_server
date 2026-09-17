"""Load the exact pre218 five-source snapshot without editing working files."""
import importlib.abc
import importlib.util
import json
import hashlib
import sys
from pathlib import Path
PACKET=Path(__file__).resolve().parent
FILES={}
for row in json.loads((PACKET/'baseline-manifest.json').read_text()):
 path=PACKET/'baseline'/row['path']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 FILES[row['path'][:-3].replace('/','.')]=path
class Loader(importlib.abc.Loader):
 def __init__(self,path): self.path=path
 def create_module(self,spec): return None
 def exec_module(self,module):
  exec(compile(self.path.read_bytes(),str(self.path),'exec'),module.__dict__)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in FILES:
   return importlib.util.spec_from_file_location(fullname,FILES[fullname],loader=Loader(FILES[fullname]))
sys.meta_path.insert(0,Finder())
