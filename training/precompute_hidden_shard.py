from __future__ import annotations
import argparse,pickle
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from training.schema_aware_training import batchify

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--shards",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=32)
    args=ap.parse_args()

    with open(ROOT/"training/cache/encoded_examples.pkl","rb") as f:
        items=pickle.load(f)
    indices=list(range(len(items)))[args.shard::args.shards]
    subset=[items[i] for i in indices]

    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"))
    model=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    model.load_state_dict(torch.load(ROOT/"tiny_gpt_schema_aware_final.pt",map_location="cpu",weights_only=True),strict=True)
    model.eval()
    pad=tok.vocab["<PAD>"]

    rows=[]
    with torch.no_grad():
        for start in range(0,len(subset),args.batch_size):
            batch=subset[start:start+args.batch_size]
            ids,roles=batchify(batch,pad)
            hidden=model.get_hidden_states(ids,roles).cpu()
            for bi,item in enumerate(batch):
                rows.append({
                    "index":indices[start+bi],
                    "hidden":hidden[bi,:len(item["input_ids"])].to(torch.float16),
                })
    path=ROOT/f"training/cache/hidden_{args.shard:02d}_of_{args.shards:02d}.pt"
    torch.save(rows,path)
    print("saved",path.name,"rows",len(rows),"bytes",path.stat().st_size)

if __name__=="__main__":main()
