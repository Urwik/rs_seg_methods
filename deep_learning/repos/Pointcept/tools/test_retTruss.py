"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
sys.path.append("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept")


from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

if __name__ == "__main__":

    exp_name = "xyznxnynz"
    sys.argv.extend(["--config-file", f"exp/retTruss/{exp_name}/crossed/config.py"])
    sys.argv.extend(["--options", f"save_path=exp/retTruss/{exp_name}/crossed", f"weight=exp/retTruss/{exp_name}/model/model_best.pth"])
    main()
