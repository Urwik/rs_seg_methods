"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
import sys
sys.path.append("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation/repos/Pointcept")

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    # args.config_file = "configs/retTruss/semseg-pt-v3m1-0-base.py"
    # args.options = "save_path=exp/scannet/semseg-pt-v2m2-0-base"
    cfg = default_config_parser(args.config_file, args.options)
    
    print(f"Options: {args.options}")
    print(f"Config: {args.config_file}")
    # exit()
    # args = default_argument_parser().parse_args()
    # cfg = default_config_parser("configs/retTruss/semseg-pt-v3m1-0-base.py", args.options)
    

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sys.argv.extend(["--config-file", "configs/retTruss/semseg-pt-v3m1-0-base.py"])
    sys.argv.extend(["--options", "save_path=exp/retTruss/xyz"])
    main()
