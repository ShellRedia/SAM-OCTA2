import argparse

def parse_args():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--adapter", type=str, default="LoDS") # LoRA, LoKR, LoCon, LoDS
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="large") # base_plus, large
    parser.add_argument("--dataset", type=str, default="3M") # 3M, 6M, ROSE, Soul
    parser.add_argument("--data_type", type=str, default="sequence") # sequence, single
    parser.add_argument("--label_type", type=str, default="Artery") # FAZ, RV, Artery, Vein
    parser.add_argument("--is_local", type=str, default="Local") # Local, Global
    parser.add_argument("--metrics", type=str, nargs='+', default=["Dice", "Jaccard", "clDice","HD95"])
    parser.add_argument("--pretrained_weight_path", type=str, default="")
    
    parser.add_argument("--model_name", type=str, default="SwinUNETR") # "SwinUNETR", "DiNTS", "SegResNet"
    parser.add_argument("--batch_size", type=int, default=4)

    # sequence
    parser.add_argument("--frame_length", type=int, nargs='+', default=[8]) 
    parser.add_argument("--prompt_frames", type=int, nargs='+', default=[2])
    parser.add_argument("--prompt_num", type=int, nargs='+', default=[2])
    parser.add_argument("--prompt_type", type=str, default="point")

    return parser.parse_args()