import os

base_dir = os.path.join('..', 'origindata', 'naja')

filenames = os.listdir(base_dir)
os.chdir(base_dir)
for item in filenames:
    new_name = item.replace('naia', 'naja')
    os.rename(item, new_name)
