import os
from glob import glob

# %%
# find labels that correspond to a speed limit sign

dataset_path = '../datasets/mtsd_speed_limits_sliced'
yaml_file = '../datasets/mtsd_speed_limits_sliced/mtsd.yaml'
new_dataset_path = '../datasets/mtsd_speed_limits_sliced_torch'

class_folder = 'speed_limit'
no_class_folder = 'no_speed_limit'

#  with open(yaml_file) as f:
#      yaml_dict = yaml.safe_load(f)
#      label_namel = yaml_dict['names']

# %%
os.system(f"mkdir {new_dataset_path}")
for t in ['train', 'val']:
    new_path = f'{new_dataset_path}/{t}'
    os.system(f"rm -rf {new_path}")
    os.system(f"mkdir {new_path}")
    os.system(f"mkdir {new_path}/{class_folder}")
    os.system(f"mkdir {new_path}/{no_class_folder}")

    images = glob(f"{dataset_path}/images/{t}/*.jpg")
    for i in images:
        if os.path.exists(i.replace('images', 'labels').replace('.jpg', '.txt')):
            os.system(f"cp {i} {new_path}/{class_folder}/{os.path.split(i)[1]}")
        else:
            os.system(f"cp {i} {new_path}/{no_class_folder}/{os.path.split(i)[1]}")
