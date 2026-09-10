from __future__ import annotations
import math,re
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

def identifier_to_text(identifier:str)->str:
    text=identifier.replace("_"," ")
    text=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",text)
    return " ".join(text.split()).lower()

class SchemaSemanticEncoderV2(nn.Module):
    def __init__(self,vocabulary_size,embedding_dimension=64,maximum_length=64,
                 number_of_heads=4,hidden_dimension=160,number_of_layers=2,pad_token_id=0):
        super().__init__()
        self.vocabulary_size=vocabulary_size;self.embedding_dimension=embedding_dimension
        self.maximum_length=maximum_length;self.number_of_heads=number_of_heads
        self.hidden_dimension=hidden_dimension;self.number_of_layers=number_of_layers
        self.pad_token_id=pad_token_id
        self.token_embedding=nn.Embedding(vocabulary_size,embedding_dimension,padding_idx=pad_token_id)
        self.position_embedding=nn.Embedding(maximum_length,embedding_dimension)
        layer=nn.TransformerEncoderLayer(d_model=embedding_dimension,nhead=number_of_heads,
            dim_feedforward=hidden_dimension,activation="gelu",batch_first=True,norm_first=True,dropout=0.0)
        self.encoder=nn.TransformerEncoder(layer,num_layers=number_of_layers)
        self.final_norm=nn.LayerNorm(embedding_dimension)
        self.pool_score=nn.Linear(embedding_dimension,1,bias=False)
        self.projection=nn.Sequential(nn.Linear(embedding_dimension,embedding_dimension),nn.GELU(),
                                      nn.Linear(embedding_dimension,embedding_dimension))
        self.role_pool_score=nn.ModuleDict({r:nn.Linear(embedding_dimension,1,bias=False) for r in ("where","order")})
        self.role_projection=nn.ModuleDict({r:nn.Sequential(nn.Linear(embedding_dimension,embedding_dimension),nn.GELU(),
            nn.Linear(embedding_dimension,embedding_dimension)) for r in ("where","order")})
        self.role_token_classifier=nn.ModuleDict({r:nn.Sequential(nn.Linear(embedding_dimension,32),nn.GELU(),
            nn.Linear(32,1)) for r in ("where","order")})
        self.log_temperature=nn.Parameter(torch.tensor(math.log(0.08),dtype=torch.float32))
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding): nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def config_dict(self):
        return {k:getattr(self,k) for k in ("vocabulary_size","embedding_dimension","maximum_length","number_of_heads",
                                            "hidden_dimension","number_of_layers","pad_token_id")}
    def encode_sequence_states(self,input_ids,attention_mask=None):
        if input_ids.dim()==1: input_ids=input_ids.unsqueeze(0)
        if attention_mask is None: attention_mask=input_ids.ne(self.pad_token_id)
        if attention_mask.dim()==1: attention_mask=attention_mask.unsqueeze(0)
        if input_ids.shape[1]>self.maximum_length:
            input_ids=input_ids[:,:self.maximum_length];attention_mask=attention_mask[:,:self.maximum_length]
        length=input_ids.shape[1]
        pos=torch.arange(length,device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0],-1)
        x=self.token_embedding(input_ids)+self.position_embedding(pos)
        x=self.encoder(x,src_key_padding_mask=~attention_mask.bool())
        return self.final_norm(x),attention_mask.bool()
    def role_token_logits(self,input_ids,attention_mask,role):
        states,mask=self.encode_sequence_states(input_ids,attention_mask)
        logits=self.role_token_classifier[role](states).squeeze(-1)
        return logits.masked_fill(~mask,-1e9),states,mask
    def forward(self,input_ids,attention_mask=None,role=None):
        states,mask=self.encode_sequence_states(input_ids,attention_mask)
        if role is None:
            logits=self.pool_score(states).squeeze(-1);projection=self.projection
        else:
            logits=self.role_token_classifier[role](states).squeeze(-1);projection=self.role_projection[role]
        logits=logits.masked_fill(~mask,-1e9)
        weights=torch.softmax(logits,dim=-1)
        pooled=(states*weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(projection(pooled),dim=-1)
    def encode_texts(self,tokenizer,texts:Iterable[str],device=None,role=None):
        texts=list(texts)
        if not texts:return torch.empty(0,self.embedding_dimension,device=device or self.token_embedding.weight.device)
        encoded=[tokenizer.encode_ids(t)[:self.maximum_length] for t in texts]
        max_len=max(max(len(x),1) for x in encoded);pad=self.pad_token_id
        rows=[];masks=[]
        for ids in encoded:
            ids=ids or [pad];n=len(ids);rows.append(ids+[pad]*(max_len-n));masks.append([1]*n+[0]*(max_len-n))
        device=device or self.token_embedding.weight.device
        return self.forward(torch.tensor(rows,dtype=torch.long,device=device),
                            torch.tensor(masks,dtype=torch.bool,device=device),role=role)
    def contrastive_loss(self,q,f):
        t=self.log_temperature.exp().clamp(0.02,0.30);logits=(q@f.T)/t
        labels=torch.arange(logits.shape[0],device=logits.device)
        return .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
    @torch.no_grad()
    def score_schema(self,tokenizer,question,schema):
        self.eval()
        q=self.encode_texts(tokenizer,[question])[0]
        qw=self.encode_texts(tokenizer,[question],role="where")[0]
        qo=self.encode_texts(tokenizer,[question],role="order")[0]
        keys=[];ctext=[];tables=list(schema);ttext=[]
        for t in tables:
            ttext.append(identifier_to_text(t))
            for c in schema[t]:keys.append((t,c));ctext.append(identifier_to_text(c))
        ce=self.encode_texts(tokenizer,ctext);te=self.encode_texts(tokenizer,ttext)
        cs=(ce@q).cpu().tolist();ws=(ce@qw).cpu().tolist();os=(ce@qo).cpu().tolist()
        ts=(te@q).cpu().tolist()
        columns={k:float(v) for k,v in zip(keys,cs)}
        where_columns={k:float(v) for k,v in zip(keys,ws)}
        order_columns={k:float(v) for k,v in zip(keys,os)}
        table_scores={}
        for t,name_score in zip(tables,ts):
            child=[columns[(t,c)] for c in schema[t]]
            best=max(child) if child else -1.0
            support=(torch.logsumexp(torch.tensor(child)/.12,0)*.12-.12*math.log(max(len(child),1))).item() if child else -1.0
            table_scores[t]=float(.70*best+.20*support+.10*name_score)
        return {"tables":table_scores,"columns":columns,"where_columns":where_columns,"order_columns":order_columns}
