from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
from src.tokenizer.bpe import BPETrainer
from pytorch_impl.tiny_gpt import TinyGPT
from text_to_sql.semantic_schema_encoder_v2 import SchemaSemanticEncoderV2
from training.semantic_pretraining_v2 import build_semantic_pairs
CHECKPOINT=ROOT/"semantic_schema_encoder_v2.pt"

def encode_batch(tok,texts,max_length,pad):
    enc=[tok.encode_ids(t)[:max_length] for t in texts]
    ml=max(max(len(x),1) for x in enc);rows=[];masks=[]
    for ids in enc:
        ids=ids or [pad];n=len(ids);rows.append(ids+[pad]*(ml-n));masks.append([1]*n+[0]*(ml-n))
    return torch.tensor(rows),torch.tensor(masks,dtype=torch.bool)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=280)
    ap.add_argument("--lr",type=float,default=.0012);ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args();random.seed(args.seed);torch.manual_seed(args.seed)
    tok=BPETrainer.load(str(ROOT/"tokenizer_balanced.json"));pad=tok.vocab["<PAD>"]
    model=SchemaSemanticEncoderV2(len(tok.vocab),64,64,4,160,2,pad)
    tiny=TinyGPT(len(tok.vocab),64,512,4,128,4,5)
    tiny.load_state_dict(torch.load(ROOT/"tiny_gpt_schema_aware_final.pt",map_location="cpu",weights_only=True),strict=True)
    with torch.no_grad():model.token_embedding.weight.copy_(tiny.token_embedding.weight)
    pairs=build_semantic_pairs(args.seed);groups=defaultdict(list)
    for x in pairs:groups[x["concept"]].append(x)
    concepts=sorted(groups);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    history=[];model.train()
    for step in range(1,args.steps+1):
        batch=[random.choice(groups[c]) for c in concepts];random.shuffle(batch)
        qids,qmask=encode_batch(tok,[x["question"] for x in batch],64,pad)
        fids,fmask=encode_batch(tok,[x["field"].replace("_"," ") for x in batch],64,pad)
        q=model(qids,qmask);f=model(fids,fmask);loss=model.contrastive_loss(q,f)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%40==0 or step==args.steps:
            with torch.no_grad():
                acc=float(((q@f.T).argmax(-1)==torch.arange(len(batch))).float().mean())
            item={"step":step,"loss":float(loss.detach()),"retrieval_accuracy":acc};history.append(item);print(json.dumps(item))
    model.eval();torch.save(model.state_dict(),CHECKPOINT)
    (ROOT/"training/semantic_encoder_v2_metrics.json").write_text(json.dumps({"history":history,"pairs":len(pairs)},indent=2))
if __name__=="__main__":main()
