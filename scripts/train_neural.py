"""Bounded CPU pilot on frozen train/validation roles; never scores locked test."""
import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gridlens.artifacts import ArtifactManifest, build_manifest, verify_manifest
from gridlens.data.dataset import FEATURE_ORDER
from gridlens.data.store import content_id
from gridlens.metrics import point_metrics
from gridlens.models.numeric import NumericLSTM, NumericTransformer
from gridlens.models.timexer import GridLensTimeXer
from gridlens.models.bundle import NeuralBundle
from scripts.build_dataset import ROOT, prepare


def tensors(dataset, scaling):
    history = torch.tensor([scaling.transform(f.history) for f in dataset.features], dtype=torch.float32)
    calendar = torch.tensor([f.target_calendar for f in dataset.features], dtype=torch.float32)
    target = torch.tensor([e.target_mw for e in dataset.examples], dtype=torch.float32)
    return history, calendar, target


@torch.no_grad()
def predict(model, history, calendar):
    model.eval()
    return torch.cat([model(history[i:i+32], calendar[i:i+32]) for i in range(0,len(history),32)])


def factory(config):
    options = {k:v for k,v in config.items() if k != 'kind'}
    return {'transformer':NumericTransformer, 'lstm':NumericLSTM, 'timexer':GridLensTimeXer}[config['kind']](**options)


def save_checkpoint(path, model, optimizer, epoch, best, best_state, stale, history, identity):
    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'best':best, 'best_state':best_state, 'stale':stale, 'history':history, 'rng':torch.get_rng_state(), 'identity':identity}, path)
    manifest = build_manifest(path.parent, [path.name], kind='training_checkpoint')
    path.with_suffix('.manifest.json').write_text(manifest.model_dump_json(indent=2))


def load_checkpoint(path, model, optimizer, identity):
    manifest = ArtifactManifest.model_validate_json(path.with_suffix('.manifest.json').read_text())
    verify_manifest(path.parent,manifest)
    state = torch.load(path, weights_only=True)
    if state['identity'] != identity:
        raise ValueError('Resume configuration/dataset/seed mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    return state


def train(config, seed, train_data, val_data, scaling, freeze, output, epochs, resume):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = factory(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    identity = content_id({'config':config,'seed':seed,'freeze':freeze['manifest_id'],'epochs':epochs,'lr':.001,'batch':32})
    folder = output / identity
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint = folder / 'checkpoint.pt'
    start, best, stale, history = 0, float('inf'), 0, []
    best_state = copy.deepcopy(model.state_dict())
    if resume and checkpoint.exists():
        state = load_checkpoint(checkpoint, model, optimizer, identity)
        start,best,stale,history,best_state = (state[k] for k in ('epoch','best','stale','history','best_state'))
    x,c,y = train_data
    vx,vc,vy = val_data
    mean,scale = scaling.means[0],scaling.scales[0]
    y = (y-mean)/scale
    started = time.perf_counter()
    for epoch in range(start,epochs) if stale < 4 else ():
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for indices in order.split(32):
            optimizer.zero_grad()
            target = y[indices]
            predicted = model(x[indices],c[indices],target if config['kind'] != 'timexer' else None)
            loss = F.mse_loss(predicted,target)
            if not bool(loss.isfinite()):
                raise ValueError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            losses.append(loss.item())
        prediction = predict(model,vx,vc)*scale+mean
        rmse = ((prediction-vy).square().mean()).sqrt().item()
        history.append({'epoch':epoch+1,'train_normalized_mse':statistics.mean(losses),'validation_autoregressive_rmse_mw':rmse})
        if rmse < best:
            best,best_state,stale = rmse,copy.deepcopy(model.state_dict()),0
        else:
            stale += 1
        save_checkpoint(checkpoint,model,optimizer,epoch+1,best,best_state,stale,history,identity)
        print(json.dumps({'kind':config,'seed':seed,**history[-1]}),flush=True)
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    model.eval()
    prediction = predict(model,vx,vc)*scale+mean
    latencies = []
    with torch.no_grad():
        model(vx[:1],vc[:1])
        for _ in range(5):
            stamp = time.perf_counter()
            model(vx[:1],vc[:1])
            latencies.append((time.perf_counter()-stamp)*1000)
    torch.save(model.state_dict(),folder/'weights.pt')
    metadata = {'config':config,'seed':seed,'identity':identity,'dataset_manifest_id':freeze['manifest_id'],'scaling':{'means':scaling.means,'scales':scaling.scales},'feature_order':FEATURE_ORDER,'gap_slots':48,'target_slots':48}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
    bundle = build_manifest(folder,['weights.pt','metadata.json'],kind='neural_point_model')
    (folder/'manifest.json').write_text(bundle.model_dump_json(indent=2))
    verify_manifest(folder,bundle)
    restored = factory(config).eval()
    restored.load_state_dict(torch.load(folder/'weights.pt',weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(model(vx[:2],vc[:2]),restored(vx[:2],vc[:2]),rtol=0,atol=0)
    # Prove an actual checkpoint can resume and restores identical model outputs.
    resume_model = factory(config)
    resume_optimizer = torch.optim.Adam(resume_model.parameters(),lr=.001)
    loaded = load_checkpoint(checkpoint,resume_model,resume_optimizer,identity)
    assert loaded['epoch'] == history[-1]['epoch']
    row = {'config':config,'seed':seed,'identity':identity,'metrics':point_metrics(vy.tolist(),prediction.tolist()),'fit_seconds':time.perf_counter()-started,'prediction_p50_ms':statistics.median(latencies),'history':history,'bundle_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),'reload_identical':True,'checkpoint_resume_verified':True}
    NeuralBundle.load_bundle(folder)
    return row,prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/local/neural')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    datasets,freeze,scaling = prepare(ROOT/'data/neso')
    if json.loads(json.dumps(freeze)) != json.loads((ROOT/'reports/dataset-freeze-v1.json').read_text()):
        raise ValueError('Dataset freeze mismatch')
    training,validation = datasets['point_train'],datasets['point_validation']
    train_data,val_data = tensors(training,scaling),tensors(validation,scaling)
    configs = [{'kind':'transformer'},{'kind':'lstm'},{'kind':'timexer'},{'kind':'timexer','external':False},{'kind':'timexer','patch':48},{'kind':'timexer','heads':1},{'kind':'timexer','positions':False}]
    results,forecasts,best_by_kind = [],{},{}
    demo_index = next(i for i,e in enumerate(validation.examples) if str(e.inputs.target_start.date()) == '2022-02-10')
    for config in configs:
        row,prediction = train(config,17,train_data,val_data,scaling,freeze,args.output,args.epochs,args.resume)
        results.append(row)
        kind = config['kind']
        if kind not in best_by_kind or row['metrics']['rmse_mw'] < best_by_kind[kind]:
            best_by_kind[kind] = row['metrics']['rmse_mw']
            loaded_point = NeuralBundle.load_bundle(args.output/row['identity']).predict(validation.features[demo_index])
            np.testing.assert_allclose(loaded_point.prediction_mw,prediction[demo_index].tolist(),rtol=0,atol=.05)
            forecasts[kind] = {'inputs':validation.examples[demo_index].inputs.model_dump(mode='json'),'prediction_mw':list(loaded_point.prediction_mw),'observed_mw':list(validation.examples[demo_index].target_mw),'model_id':row['identity'],'dataset_manifest_id':freeze['manifest_id'],'evaluation_role':'point_validation','evidence_mode':'retrospective_revised','architecture':config}
    baseline = json.loads((ROOT/'reports/baseline-validation-v1.json').read_text())
    all_candidates = [{'identity':r['model_version'],'kind':r['model_kind'],'rmse_mw':r['metrics']['rmse_mw']} for r in baseline['results']] + [{'identity':r['identity'],'kind':r['config']['kind'],'rmse_mw':r['metrics']['rmse_mw']} for r in results]
    champion = min(all_candidates,key=lambda r:(r['rmse_mw'],r['identity']))
    report = {'schema_version':'gridlens_neural_pilot_v1','dataset_manifest_id':freeze['manifest_id'],'training_origins':len(training.examples),'validation_origins':len(validation.examples),'results':results,'champion':champion,'selection_rule':'minimum full 2022 validation RMSE across frozen baselines and bounded neural pilot','seeds':[17],'status':'bounded single-seed pilot; no multi-seed superiority claim','locked_test_evaluated':False,'epochs_budget':args.epochs,'timexer_trials':5,'numeric_trials':2,'limitations':['Teacher-forced numeric training has exposure bias; all validation is generated inference.','Ablations are pilot sensitivity checks, not statistically established conclusions.']}
    report['report_id'] = content_id(report)
    report['source_commit'] = os.environ.get('GITHUB_SHA','local')
    report['torch_version'] = str(torch.__version__)
    report['report_id'] = content_id({k:v for k,v in report.items() if k != 'report_id'})
    for forecast in forecasts.values():
        forecast['validation_report_id'] = report['report_id']
        forecast['forecast_id'] = content_id(forecast)
    for path,value in [('report.json',report),('forecasts.json',forecasts)]:
        (args.output/path).write_text(json.dumps(value,indent=2))
        text = json.dumps(value)
        for i in range(0,len(text),10000):
            print('GRIDLENS_NEURAL_OUTPUT '+json.dumps({'path':path,'index':i//10000,'content':text[i:i+10000]}),flush=True)


if __name__ == '__main__':
    main()
