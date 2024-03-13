import os
import tempfile

temp_dir = tempfile.TemporaryDirectory()
with open(temp_dir.name + "/yo.txt", "w") as o:
    pass
print(temp_dir.name, os.listdir(temp_dir.name))
raise Exception("STOP...")
# use temp_dir, and when done:
temp_dir.cleanup()