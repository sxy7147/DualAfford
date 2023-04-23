import json

cs_file = open("./cate_setting.json", "w")
cate_field_name = ["obj_scale", "urdf", "joint_range", "base_theta", "restrict_dir", "low_view", "lie_down", "RL_exp_num"]
cate_setting = {
    "shape_id": {
        "Eyeglasses": "01",     "Bucket": "02",     "Box": "03",            "Cart": "04",       "Chair": "05",
        "TrashCan": "06",       "Pliers": "07",     "FoldingChair": "08",   "Basket": "09",     "Display": "10",
        "Table": "11",          "Oven": "12",       "Dishwasher": "13",     "KitchenPot": "14", "Microwave": "15",
        "Scissors": "16",       "Laptop": "17",     "BirdHouse": "18",      "Skateboard": "19", "Bookshelf": "20",
        "Safe": "21",           "Toaster": "22",    "Printer": "23",        "Bench": "24",      "Keyboard2": "25"
    },
    "pickup": {
        "cate_setting": {
            "Eyeglasses":   [0.90, "processed", (0, 0.3), 1.0, True, False, False, "24000"],
            "Bucket":       [0.85, "removed", (0.99, 1.0), 1.0, True, False, False, "24000"],
            "TrashCan":     [0.80, "removed", (0.45, 0.55), 1.0, True, False, False, "24000"],
            "Pliers":       [0.75, "processed", (0.3, 1.0), 1.0, True, False, False, "24000"],
            "Basket":       [0.75, "shapenet", (0.0, 1.0), 1.0, True, False, False, "24000"],
            "Display":      [0.80, "origin", (0, 0.03), 1.0, True, False, False, "24000"],

            "KitchenPot":   [0.68, "removed", (0.0, 1.0), 1.0, True, False, False, "35000"],
            "Box":          [0.80, "removed", (0.0, 0.05), 1.0, True, False, False, "52000"],
            "Scissors":     [0.75, "fixbase", (0.3, 1.0), 1.0, True, False, False, "36000"],
            "Laptop":       [0.80, "origin", (0.0, 0.0), 1.0, True, False, False, "39000"]
        },
        "material_static_friction": 8.0,
        "material_dynamic_friction": 8.0,
        "scene_static_friction": 0.60,
        "scene_dynamic_friction": 0.60,
        "stiffness": 1000,
        "damping": 10,
        "density": 9.0,
        "gripper_scale": 1.65,
        "start_dist": 0.40,
        "final_dist": 0.10,
        "move_steps": 4000,
        "wait_steps": 2000,
        "RL_load_date": "0311",
        "finetune_path": "../2gripper_logs/finetune/exp-finetune-CVPR_baseline-pickup-Eyeglasses,Bucket,TrashCan,Pliers,Basket,Display-050980",
        "finetune_eval_epoch": "2-300"
    },
    "pulling": {
        "cate_setting": {
            "Eyeglasses":   [0.80, "processed", (0, 0.3), 1.0, True, False, False, "120000"],
            "FoldingChair": [0.80, "processed", (0.0, 0.1), 1.0, True, False, False, "27000"],
            "Oven":         [0.75, "processed", (0.95, 1.0), 1.0, True, False, False, "XXXXX"],
            "Dishwasher":   [0.75, "processed", (0.95, 1.0), 1.0, True, False, False, "83000"],
            "Skateboard":   [0.80, "shapenet", (0.95, 1.0), 1.0, True, True, False, "113300"],
            "Bookshelf":    [0.75, "shapenet", (0.95, 1.0), 0.5, True, False, False, "161000"],

            "Table":        [0.80, "removed", (0.0, 0.01), 1.0, True, True, False, "86000"],
            "Box":          [0.80, "removed", (0.0, 0.05), 1.0, True, True, True, "136000"],
            "Safe":         [0.75, "processed", (0.95, 1.0), 1.0, True, False, False, "XXXXX"],
            "Microwave":    [0.75, "processed", (0.95, 1.0), 1.0, True, False, False, "XXXXX"],
            "BirdHouse":    [0.75, "shapenet", (0.95, 1.0), -0.5, True, False, False, "XXXXX"]
        },
        "material_static_friction": 8.0,
        "material_dynamic_friction": 8.0,
        "scene_static_friction": 0.25,
        "scene_dynamic_friction": 0.25,
        "stiffness": 1000,
        "damping": 10,
        "density": 8.0,
        "gripper_scale": 1.50,
        "start_dist": 0.40,
        "final_dist": 0.10,
        "move_steps": 4000,
        "wait_steps": 1500,
        "RL_load_date": "0420"
    },
    "pushing": {
        "cate_setting": {
            "Toaster":      [0.75, "origin", (0.0, 0.01), 1.0, True, False, False, "XXXXX"],
            "KitchenPot":   [0.75, "origin", (0.0, 0.01), 1.0, True, False, False, "XXXXX"],
            "Basket":       [0.75, "shapenet", (0.0, 0.01), 1.0, True, False, False, "XXXXX"],

            "Box":          [1.00, "origin", (0.0, 0.01), 1.0, True, False, False, "19500"],
            "Dishwasher":   [0.75, "origin", (0.0, 0.01), 1.0, True, False, False, "52400"],
            "Display":      [1.00, "origin", (0.0, 0.01), 1.0, True, False, False, "58300"],
            "Microwave":    [1.00, "origin", (0.0, 0.01), 1.0, True, False, False, "XXXXX"],
            "Printer":      [1.00, "origin", (0.0, 0.01), 1.0, True, False, False, "XXXXX"],
            "Bench":        [1.00, "shapenet", (0.0, 0.01), 1.0, True, False, False, "65700"],
            "Keyboard2":    [1.00, "shapenet", (0.0, 0.01), 1.0, True, False, False, "29600"],
        },
        "material_static_friction": 4.0,
        "material_dynamic_friction": 4.0,
        "scene_static_friction": 0.30,
        "scene_dynamic_friction": 0.30,
        "stiffness": 1,
        "damping": 10,
        "density": 2.0,
        "gripper_scale": 2,
        "start_dist": 0.45,
        "final_dist": 0.10,
        "move_steps": 3500,
        "wait_steps": 2000,
        "RL_load_date": "1105",
        "finetune_path": "../2gripper_logs/finetune/exp-finetune-CVPR_baseline-pushing-Box,Bucket,Dishwasher,Display,Microwave,Bench,Bowl,Keyboard2-042881",
        "finetune_eval_epoch": "8-0"
    },
    "rotating": {
        "finetune_path": "../2gripper_logs/finetune/exp-finetune-CVPR_baseline-rotating-Box,Bucket,Dishwasher,Display,Microwave,Bench,Bowl,Keyboard2-051480",
        "finetune_eval_epoch": "16-0",
        "scene_static_friction": 0.15,
        "scene_dynamic_friction": 0.15,
    },
    "topple": {
        "finetune_path": "../2gripper_logs/finetune/exp-finetune-CVPR_baseline-topple-Box,Bucket,Dishwasher,Display,Bench,Bowl-050580",
        "finetune_eval_epoch": "9-0",
        "scene_static_friction": 0.15,
        "scene_dynamic_friction": 0.15,
    },
    "fetch": {
        "material_static_friction": 8.0,
        "material_dynamic_friction": 8.0,
        "scene_static_friction": 0.60,
        "scene_dynamic_friction": 0.60,
        "stiffness": 1,
        "damping": 10,
        "density": 80,
        "gripper_scale": 1.0,
        "start_dist": 0.50,
        "final_dist": 0.10,
        "move_steps": 5000,
        "wait_steps": 2000,
    }
}

for primact in cate_setting:
    if primact != "shape_id":
        if "cate_setting" in cate_setting[primact]:
            for cate in cate_setting[primact]["cate_setting"]:
                cate_setting[primact]["cate_setting"][cate] = dict(zip(cate_field_name, cate_setting[primact]["cate_setting"][cate]))

json.dump(cate_setting, cs_file)
