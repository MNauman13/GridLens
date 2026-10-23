"""Export an independent calculator fixture, verified against manual PyTorch operators."""
import argparse
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TOKENS=[[1.,0.,0.,1.],[0.,1.,2.,0.],[1.,1.,1.,1.]]

def rounded(value):
    if isinstance(value,float): return round(value,9)
    if isinstance(value,list): return [rounded(v) for v in value]
    if isinstance(value,dict): return {k:rounded(v) for k,v in value.items()}
    return value

def reference():
    heads=[[t[i:i+2] for t in TOKENS] for i in (0,2)]
    scenarios={}
    for name,causal in (("unmasked",False),("causal",True)):
        scores,weights,outputs=[],[],[]
        for head in heads:
            matrix=[[sum(q*k for q,k in zip(query,key,strict=True))/math.sqrt(2) for key in head] for query in head]
            normalised=[]
            for query,row in enumerate(matrix):
                allowed=[i for i in range(3) if not causal or i<=query]
                maximum=max(row[i] for i in allowed)
                exp=[math.exp(row[i]-maximum) if i in allowed else 0. for i in range(3)]
                normalised.append([v/sum(exp) for v in exp])
            scores.append(matrix)
            weights.append(normalised)
            outputs.append([[sum(row[i]*head[i][c] for i in range(3)) for c in range(2)] for row in normalised])
        scenarios[name]={"scores":scores,"weights":weights,"head_outputs":outputs,"merged_output":[outputs[0][i]+outputs[1][i] for i in range(3)]}
    positions=[[math.sin(p/10000**(2*(c//2)/8)) if c%2==0 else math.cos(p/10000**(2*(c//2)/8)) for c in range(8)] for p in range(16)]
    return rounded({"schema_version":"attention_teaching_fixture_v1","tokens":TOKENS,"heads":heads,"scenarios":scenarios,"positions":positions,"projection_policy":"identity Q,K,V and output projections; four channels split into two heads","mask_convention":"True means blocked"})

def verify(data):
    import torch
    from gridlens.models.attention import MultiHeadAttention,SinusoidalPositionalEncoding,causal_mask
    model=MultiHeadAttention(4,2).double().eval()
    with torch.no_grad():
        for layer in (model.q_proj,model.k_proj,model.v_proj,model.out_proj):
            layer.weight.copy_(torch.eye(4,dtype=torch.float64))
            layer.bias.zero_()
    inputs=torch.tensor(TOKENS,dtype=torch.float64).unsqueeze(0)
    for name in ("unmasked","causal"):
        output,weights=model(inputs,blocked=causal_mask(3) if name=="causal" else None)
        torch.testing.assert_close(weights[0],torch.tensor(data["scenarios"][name]["weights"],dtype=torch.float64),atol=1e-8,rtol=1e-8)
        torch.testing.assert_close(output[0],torch.tensor(data["scenarios"][name]["merged_output"],dtype=torch.float64),atol=1e-8,rtol=1e-8)
    pe=SinusoidalPositionalEncoding(8,16)(torch.zeros(1,16,8))[0]
    torch.testing.assert_close(pe,torch.tensor(data["positions"],dtype=torch.float32),atol=1e-6,rtol=1e-6)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    data=reference()
    path=ROOT/"apps/web/data/attention-demo.json"
    if args.check and json.loads(path.read_text())!=data: raise SystemExit("Attention fixture is stale")
    verify(data)
    if not args.check: path.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print("Teaching attention and positional values match manual PyTorch modules")
if __name__=="__main__": main()
