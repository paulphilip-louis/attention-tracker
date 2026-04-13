import sys, argparse
from transformer_lens import HookedTransformer
from utils import utils
import json


def find_heads():
    print("Function started")
    parser = argparse.ArgumentParser(
                    prog='Attention-tracker',
                    description='Detection of prompt injection from attention patterns',
                    )
    print("Parser: done")
    parser.add_argument("model_name", type=str, help="Huggingface name of the model")
    parser.add_argument("--k", nargs='?', default=4, type=int, help="Hyperparameter used for selection of heads. Higher k means fewer heads. Default=4")
    parser.add_argument('-p', '--plot', action='store_true', help='Plots a figure')
    print("Parsing the arguments")
    args = parser.parse_args()
    print("Loading the model!")
    model = HookedTransformer.from_pretrained(args.model_name)
    print("Building dataset!!")
    normal_dataset, injected_dataset = utils.generate_dataset()
    print("FINDING IMPORTANT HEADS!!!")
    heads = utils.find_important_heads(model, normal_dataset, injected_dataset, k=args.k)
    print(heads)

def run_benchmark():
    parser = argparse.ArgumentParser(
                    prog='Attention-tracker',
                    description='Detection of prompt injection from attention patterns',
                    )
    parser.add_argument("model_name", type=str, help="Huggingface name of the model")
    parser.add_argument("heads", type=str, help="filename of important heads")
    parser.add_argument("benchmark_name", type=str)
    parser.add_argument("threshold", type=float)
    args = parser.parse_args()
    model = HookedTransformer.from_pretrained(args.model_name)
    heads = json.load(open(args.heads))
    auroc, accuracy = utils.run_on_benchmark(model, heads, args.threshold, args.benchmark_name)
    sys.stdout.write("######################################")
    sys.stdout.write(f"Results on {args.benchmark_name} :")
    sys.stdout.write(f"AUROC : {auroc:.3f}")
    sys.stdout.write(f"Accuracy : {100*accuracy:.1f}%")

def detect():
    parser = argparse.ArgumentParser(
                    prog='Attention-tracker',
                    description='Detection of prompt injection from attention patterns',
                    )
    parser.add_argument("model_name", type=str, help="Huggingface name of the model")
    parser.add_argument("heads", type=str, help="filename of important heads")
    parser.add_argument("instruction", type=str, help="instruction")
    parser.add_argument("data", type=str, help="data")
    parser.add_argument("threshold", type=float)
    args = parser.parse_args()
    model = HookedTransformer.from_pretrained(args.model_name)
    heads = json.load(args.heads)
    prompt = utils.Prompt(instruction=args.instruction, data=args.data)
    fs = utils.focus_score(model, heads, prompt)
    if fs > args.threshold:
        sys.stdout.write("Safe")
    else:
        sys.stdout.write("Danger! Prompt injection detected")