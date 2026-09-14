import os
import sys
import itertools
import argparse
import yaml
import re
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."+".", "."+"."))
sys.path.insert(0, parent_dir)

from script.helper import get_parameter_settings, parse_sub_runs

from utils import ImageType, RewardType

DATE =  os.path.dirname(__file__).split("/")[-1] # "2025_12_24"
EXP_ID = int(re.search(r'\d+', os.path.splitext(os.path.basename(__file__))[0]).group()) # 0
ABOUT = "RL for Seq DOE with subset of angles"

def setup_setting_files(seed, time_limit, print_info, skip_save=False):
    od = get_parameter_settings(seed, False, ABOUT)

    od["seed"] = seed
    od["time_limit"] = time_limit

    image_type_arr = [ImageType.TRIANGLE, ImageType.TRIANGLE_FIXED]
    reward_type_arr = [RewardType.FWD_E2E, RewardType.FWD_INCREMENTAL]
    n_angles_arr = [6,24,96]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not skip_save:
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)
        if not(os.path.exists(setting_folder_base)):
            os.makedirs(setting_folder_base)
        print("Saving setting files to %s" % setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "image_type", "reward_type", "n_angles"]
    row_format ="{:>10}|{:>15}|{:>15}|{:>10}"
    if not skip_save:
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (50+len(exp_metadata)-1))

    ct = 0
    for (image_type, reward_type, n_angles) in itertools.product(image_type_arr, reward_type_arr, n_angles_arr):
        od["image_type"] = image_type.value
        od["reward_type"] = reward_type.value
        od["n_angles"] = n_angles

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        if not skip_save:
            print(row_format.format(ct, image_type.name, reward_type.name, od["n_angles"]))

            if not(os.path.exists(od["log_folder"])):
                os.makedirs(od["log_folder"])
            with open(setting_fname, 'w') as f:
                # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
                yaml.dump(od, f, default_flow_style=False, sort_keys=False)
        ct += 1

    return ct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--run", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--print_info", action="store_true", help="Prints description of all settings")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="single", 
        choices=["test", "single"],
        help="Set up number of trials and max_step for various testing reasons"
    )
    parser.add_argument(
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    parser.add_argument("--parallel", action="store_true", help="Run seeds in parallel")

    args = parser.parse_args()
    seed = 1029
    time_limit = 1_800

    if args.setup:
        if args.mode == "test":
            time_limit = 30
        setup_setting_files(seed, time_limit, args.print_info)
    elif args.run:
        max_runs = setup_setting_files(seed, time_limit, args.print_info, True)
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs, max_runs)
        folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)

        for i in range(start_run_id, end_run_id):
            settings_file = os.path.join(folder_name, "run_%i.yaml" % i)
            os.system('echo "Running exp id %d"' % i)
            os.system("python Actor_Critic.py --settings %s%s" % (
                settings_file, 
                " --parallel" if args.parallel else "",
            ))
    else:
        print("Neither setup nor run passed. Shutting down...")

