import argparse

def parse_args():    
    parser = argparse.ArgumentParser()

    parser.add_argument("-epochs", type=int, default=50, help="")
    parser.add_argument("-model_type", type=str, default="large") # base_plus
    parser.add_argument("-task_type", type=str, default="projection") # layer_sequence, projection
    parser.add_argument("-label_type", type=str, default="FAZ")
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard"])

    return parser.parse_args()