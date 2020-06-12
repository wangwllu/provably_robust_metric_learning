import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fashion')
    parser.add_argument('--root', default='../../Datasets/pytorch')
    parser.add_argument('--split_seed', type=int, default=42)
    