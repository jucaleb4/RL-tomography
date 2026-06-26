import math

def parse_sub_runs(sub_runs, total_runs):
    start_run_id, end_run_id = 0, total_runs
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= total_runs, "sub_runs id must be in [0,%s]" % (total_runs-1)
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % (total_runs-1))

    return start_run_id, end_run_id

def get_parameter_settings(seed, print_info, about):
    od = dict([
        ("log_folder", ""),
        ("seed", seed), 
        ("n_episodes", 300_000),
		("time_limit", 3600),
        ("n_images", 3000), # number of sampled images
        ("n_angles", 6), # number of angles to save
        ("reward_type", "forward"), # forward or pnsr
        ("gamma", 0.99),
        ("lr", 1e-4),
        ("wd", 1e-5), # weight decay
        ("delta", 1e-2),
        ("image_size", 128), # TODO: remove this?
        ("action_size", 180), # TODO: remove this?
    ])

    od_info = [
        ("log_folder", "Log folder"),
        ("seed_0", "start seed"), 
        ("n_seeds", "num seeds"), 
        ("n_iters", "num SPMD iters"),
        ("max_runtime_in_sec", "max runtime before SPMD early terminates (only for SPMD)"),
    ]

    if print_info:
        print("About:\n\t%s" % about)
        exp_metadata = ["setting", "description"]
        row_format ="{:<20}|{:<60}"
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (80+len(exp_metadata)-1))
        for name, description in od_info:
            print(row_format.format(name, description))
        print("-" * (80+len(exp_metadata)-1))

    return od

