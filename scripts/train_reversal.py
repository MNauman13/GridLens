"""Bounded, seeded sequence reversal reproduction; no translation claims."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from gridlens.models.transformer import TokenTransformer, warmup_rate


def batch(rng, count, low=3, high=8):
    sequences = [[rng.randrange(3, 13) for _ in range(rng.randint(low, high))] for _ in range(count)]
    length = max(map(len, sequences)) + 1
    source = torch.zeros(count, length, dtype=torch.long)
    target = torch.zeros(count, length + 1, dtype=torch.long)
    for i, seq in enumerate(sequences):
        source[i, :len(seq) + 1] = torch.tensor(seq + [2])
        target[i, :len(seq) + 2] = torch.tensor([1] + seq[::-1] + [2])
    return source, target


def accuracy(model, source, target):
    prediction = model.generate(source, max_tokens=target.shape[1] + 2)
    length = max(prediction.shape[1], target.shape[1])
    prediction = F.pad(prediction, (0, length - prediction.shape[1]))
    target = F.pad(target, (0, length - target.shape[1]))
    return (prediction == target).all(-1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--output', default='reports/local/reversal')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = TokenTransformer(dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    held_source, held_target = batch(random.Random(101), 256)
    long_source, long_target = batch(random.Random(102), 128, 9, 12)
    rng = random.Random(17)
    started = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        source, target = batch(rng, 64)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(source, target[:, :-1]).reshape(-1, 13), target[:, 1:].reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in optimizer.param_groups:
            group['lr'] = warmup_rate(step, 128, 200)
        optimizer.step()
        if step % 200 == 0:
            model.eval()
            row = {'step': step, 'loss': loss.item(), 'exact_accuracy': accuracy(model, held_source, held_target)}
            history.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'weights.pt')
    clone = TokenTransformer().eval()
    clone.load_state_dict(torch.load(output / 'weights.pt', weights_only=True))
    torch.testing.assert_close(model(held_source[:2], held_target[:2, :-1]), clone(held_source[:2], held_target[:2, :-1]), rtol=0, atol=0)
    report = {'seed': 17, 'steps': args.steps, 'architecture': {'width':128,'heads':4,'layers':2,'hidden':512}, 'held_out_sequences':256, 'exact_accuracy':accuracy(model,held_source,held_target), 'longer_length_exact_accuracy':accuracy(model,long_source,long_target), 'seconds':time.perf_counter()-started,'history':history,'reload_identical':True,'test_set_used':False}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print('GRIDLENS_REVERSAL_REPORT ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
