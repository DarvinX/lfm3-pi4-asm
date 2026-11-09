#!/usr/bin/env python3
"""H34 offline residual-bound traffic screen; no inference row is skipped.

Two saved native vectors only. Approximate scores never choose an unchecked
output. Formal rounding proof and real assembly overhead remain requirements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare import Reader,EXPECTED
from vocab_bound_feasibility import q8_pack
ROOT=Path(__file__).resolve().parents[1]


def codes(raw):
    q=np.empty((*raw.shape[:2],256),dtype='i2')
    for half in range(2):
        hi=raw[:,:,128+half*32:160+half*32]
        for group in range(4):
            lo=raw[:,:,64*half+(group%2)*32:64*half+(group%2)*32+32]
            lo=(lo&15) if group<2 else lo>>4
            q[:,:,128*half+group*32:128*half+(group+1)*32]=(lo|(((hi>>(group*2))&3)<<4)).astype('i2')-32
    scales=np.repeat(raw[:,:,192:208].view('i1').astype('i2'),16,axis=2)
    d=raw[:,:,208:210].copy().view('<f2').reshape(raw.shape[:2]).astype('f4')
    return q,scales,d


def logits(coeff,q,alpha):
    integer=np.einsum('rbk,bk->rb',coeff.astype('i4'),q,optimize=False)
    acc=np.zeros(len(coeff),dtype='f4')
    for b in range(8):
        acc=(acc.astype('f8')+alpha[:,b].astype('f8')*integer[:,b].astype('f4').astype('f8')).astype('f4')
    return acc


def touched_bytes(mask,start,rowbytes,unit):
    selected=np.flatnonzero(mask).astype('i8')
    first=(start%unit)+selected*rowbytes
    count=(first%unit+rowbytes+unit-1)//unit
    slots=np.arange((rowbytes+2*unit-2)//unit,dtype='i8')
    touched=np.zeros((start%unit+len(mask)*rowbytes+unit-1)//unit,dtype=bool)
    addresses=first[:,None]//unit+slots[None,:]
    touched[addresses[slots[None,:]<count[:,None]]]=True
    return int(touched.sum())*unit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'research/h34')
    a=p.parse_args();model=Reader(a.model)
    digest=hashlib.sha256(model.data).hexdigest();assert digest==EXPECTED
    off,shape,kind=model.tensors['token_embd.weight'];assert shape==[2048,128000] and kind==14
    start=model.base+off;rows=128000;rowbytes=1680
    raw=np.frombuffer(model.data,dtype='u1',count=rows*rowbytes,offset=start).reshape(rows,8,210)
    cases=[]
    for label,folder,norm in [('Hello','trace-h9-hello','01051-result_norm.f32'),('Hello world!','trace-h9-multi','03157-result_norm.f32')]:
        folder=ROOT/'native-h9'/folder/'llama';normpath=folder/norm
        ref=np.fromfile(folder/'logits.f32',dtype='<f4');x=np.fromfile(normpath,dtype='<f4')
        assert len(ref)==rows and len(x)==2048
        da,q=q8_pack(x)
        qnorm=np.nextafter(np.sqrt(np.sum(q.astype('i8')**2,axis=1).astype('f8')),np.inf)
        cases.append({'prompt':label,'norm_sha256':hashlib.sha256(normpath.read_bytes()).hexdigest(),
                      'reference':ref,'q':q,'da':da,'qnorm':qnorm,'original':np.zeros(rows,dtype='f4'),
                      'approx':{bits:np.zeros(rows,dtype='f4') for bits in [3,4,5]},
                      'bound':{bits:np.zeros(rows,dtype='f8') for bits in [3,4,5]}})
    for first in range(0,rows,1024):
        code,scale,dw=codes(raw[first:first+1024]);n=len(code)
        original=code*scale
        coarse={};norms={}
        for bits in [3,4,5]:
            step=1<<(6-bits)
            coarse[bits]=(((code+32)//step)*step+step//2-32)*scale
            residual=(original-coarse[bits]).astype('f8')
            norm=np.sqrt(np.sum(residual**2,axis=2))
            # Stored float32 norms are rounded UP, never nearest-only.
            norms[bits]=np.nextafter(norm.astype('f4'),np.float32(np.inf)).astype('f8')
            assert np.all(norms[bits]>=norm)
        for case in cases:
            alpha=np.asarray(dw*case['da'],dtype='f4')
            case['original'][first:first+n]=logits(original,case['q'],alpha)
            # For |integer dot|<=2^27, SCVTF error<=8 and8 FMA steps,
            # 160*sum|alpha| exceeds the combined first-order RN error.
            # Extra1e-3 is conservative screening slack, NOT a proof claim.
            rounding=160*np.sum(np.abs(alpha.astype('f8')),axis=1)+1e-3
            for bits in [3,4,5]:
                case['approx'][bits][first:first+n]=logits(coarse[bits],case['q'],alpha)
                error=np.sum(np.abs(alpha.astype('f8'))*norms[bits]*case['qnorm'],axis=1)
                case['bound'][bits][first:first+n]=error*(1+2**-40)+rounding
    results=[]
    for case in cases:
        ref=case['reference'];assert np.array_equal(ref.view('u4'),case['original'].view('u4')),'original oracle is not native-bit-exact'
        winner=int(ref.argmax())
        for bits in [3,4,5]:
            approx=case['approx'][bits];bound=case['bound'][bits]
            upper=approx.astype('f8')+bound
            assert np.all(np.abs(ref.astype('f8')-approx.astype('f8'))<=bound),'screen error bound violated'
            coarse_winner=int(approx.argmax());initial=float(ref[coarse_winner])
            for threshold_kind,threshold in [('achievable_coarse_winner',initial),('known_best_diagnostic',float(ref[winner]))]:
                mask=upper>=threshold;mask[coarse_winner]=True
                assert mask[winner]
                original_bytes=touched_bytes(mask,start,rowbytes,64)
                original_resident=touched_bytes(mask,start,rowbytes,4096)
                compact=rows*8*(256*bits//8+18)
                norms_bytes=rows*8*4
                saved=rows*rowbytes-compact-norms_bytes-original_bytes
                results.append({'prompt':case['prompt'],'bits':bits,'threshold_kind':threshold_kind,
                    'norm_sha256':case['norm_sha256'],'native_logit_bit_differences':0,
                    'true_top_id':winner,'coarse_top_id':coarse_winner,'initial_exact_threshold':initial,
                    'true_top_score':float(ref[winner]),'refine_rows':int(mask.sum()),
                    'rows_filtered_percent':100*(1-float(mask.mean())),
                    'compact_output_bytes':compact,'residual_norm_metadata_bytes':norms_bytes,
                    'original_refine_cacheline_bytes':original_bytes,
                    'estimated_steady_output_resident_bytes':compact+norms_bytes+original_resident,
                    'net_whole_model_traffic_saving_percent':100*saved/1089828864,
                    'max_abs_approx_logit_error':float(np.max(np.abs(ref.astype('f8')-approx.astype('f8')))),
                    'min_bound_minus_actual_error':float(np.min(bound-np.abs(ref.astype('f8')-approx.astype('f8'))))})
    actual=[r for r in results if r['threshold_kind']=='achievable_coarse_winner']
    perbits={str(b):{'mean_filtered_percent':float(np.mean([r['rows_filtered_percent'] for r in actual if r['bits']==b])),
                    'mean_net_traffic_saving_percent':float(np.mean([r['net_whole_model_traffic_saving_percent'] for r in actual if r['bits']==b])),
                    'each_case_at_least4_percent_saving':all(r['net_whole_model_traffic_saving_percent']>=4 for r in actual if r['bits']==b)} for b in [3,4,5]}
    summary={'scope':'offline traffic feasibility only; no runtime or native speed claim','source_sha256':digest,
             'output_weight_bytes':rows*rowbytes,'original_native_logits_bit_exact':True,'by_bits':perbits,
             'prediction_some3or4_mean_filter_ge90_and_net_ge4':any(perbits[str(b)]['mean_filtered_percent']>=90 and perbits[str(b)]['mean_net_traffic_saving_percent']>=4 for b in [3,4]),
             'prototype_eligible_bits':[b for b in [3,4] if perbits[str(b)]['each_case_at_least4_percent_saving']],
             'limitations':['Only two saved native query vectors.','F64-emulated coarse scores have no general FMA proof yet.',
                            'Conservative analytic/extra slack is not a formal certificate.','No kernel/classifier/gather/metadata/checkpoint-validation cost included.',
                            'Resident-byte estimates are not peakRSS; initial original integrity scan remains.']}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
