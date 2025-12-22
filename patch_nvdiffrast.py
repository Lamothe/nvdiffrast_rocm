import os
import glob

# Find where 'antialias.cu' or 'antialias.hip' actually lives
search_files = ['antialias.cu', 'antialias.hip']
found_path = None
found_ext = None

print("Searching for source files...")
for root, dirs, files in os.walk("."):
    for file in files:
        if file in search_files:
            found_path = os.path.join(root, file)
            found_ext = file.split('.')[-1]
            print(f"FOUND source: {found_path}")
            break
    if found_path: break

if not found_path:
    print("CRITICAL ERROR: Could not find antialias.cu or .hip anywhere!")
    exit(1)

# Determine the directory relative to setup.py
rel_dir = os.path.dirname(found_path).replace("./", "")
print(f"Correct source directory appears to be: '{rel_dir}'")

# Read setup.py
with open("setup.py", "r") as f:
    content = f.read()

# Brutal Patch: We replace the hardcoded sources list with the correct one
# Most setup.py files here list sources like 'csrc/common/antialias.cu'
# We will fix the path string.
if "csrc/common/antialias" in content:
    print("Patching incorrect 'csrc/common' paths...")
    # Replace the bad path prefix with the one we found
    new_content = content.replace("csrc/common/antialias", f"{rel_dir}/antialias")
    new_content = new_content.replace("csrc/common/rasterize", f"{rel_dir}/rasterize")
    new_content = new_content.replace("csrc/common/interpolate", f"{rel_dir}/interpolate")
    new_content = new_content.replace("csrc/common/texture", f"{rel_dir}/texture")
    
    # Also handle extension mismatch (.cu vs .hip)
    if found_ext == "hip":
        new_content = new_content.replace(".cu", ".hip")
    
    with open("setup.py", "w") as f:
        f.write(new_content)
    print("setup.py patched successfully!")
else:
    print("setup.py didn't match expected pattern. Attempting manual forced patch.")
    # Fallback: Just replace .cu with .hip if that was the only issue
    if found_ext == "hip":
        content = content.replace(".cu", ".hip")
        with open("setup.py", "w") as f:
            f.write(content)
