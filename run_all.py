import subprocess

# tasks = [
#     ["--frame_length", "4", "8", "12"],
#     ["--prompt_frames", "1", "2", "3"],
#     ["--prompt_num", "1", "2", "3", "4"]
# ]

    
# for args in tasks:
#     subprocess.run(["python", "train_sam_octa2.py"] + args)



tasks = []

for is_local in ["Local", "Global"]:
    tasks.append(["--is_local", is_local])


for args in tasks:
    subprocess.run(["python", "train_sam_octa2.py"] + args)

# tasks = []
# for model_name in ["SwinUNETR"]: # "SwinUNETR", "DiNTS", "SegResNet"
#     for label_type in ["RV", "Artery", "Vein", "FAZ"]: #"RV", "Artery",
#         for fov in ["3M", "6M"]:
#             tasks.append(["--model_name", model_name, "--label_type", label_type, "--dataset", fov])
# for args in tasks:
#     subprocess.run(["python", "train_monai.py"] + args)
