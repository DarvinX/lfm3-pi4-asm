#!/usr/bin/env python3
"""Optimistic offline vocabulary-pruning screen, never used by inference.

The true final best logit is supplied in advance. Norm bounds get a generous
1e-3 floating guard, but this is NOT a formal floating-point certificate.
Native adoption would need one, plus real best-so-far/overhead measurements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from prepare import Reader, EXPECTED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from native_math_reference import native_round


def q8_pack(x):
    q=np.zeros((len(x)//256,256),dtype='i4')
    d=np.zeros(len(x)//256,dtype='f4')
    for b,v in enumerate(x.reshape(-1,256)):
        maximum=v[np.argmax(np.abs(v))]
        if maximum==0:
            continue
        scale=np.float32(np.float32(-127)/maximum)
        d[b]=np.float32(np.float32(1)/scale)
        q[b]=[native_round(scale,z) for z in v]
    assert np.all((q>=-128)&(q<=127))
    return d,q


def decode(raw):
    codes=np.empty((*raw.shape[:2],256),dtype='i2')
    for half in range(2):
        qh=raw[:,:,128+32*half:160+32*half]
        for group in range(4):
            ql=raw[:,:,64*half+(group%2)*32:64*half+(group%2)*32+32]
            low=(ql&15) if group<2 else (ql>>4)
            code=low|(((qh>>(group*2))&3)<<4)
            codes[:,:,128*half+group*32:128*half+(group+1)*32]=code.astype('i2')-32
    scales=raw[:,:,192:208].view('i1').astype('i2')
    coefficients=codes*np.repeat(scales,16,axis=2)
    d=raw[:,:,208:210].copy().view('<f2').reshape(raw.shape[:2]).astype('f4')
    return coefficients,d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'research/h28')
    a=p.parse_args()
    with a.model.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==EXPECTED
    model=Reader(a.model)
    offset,shape,kind=model.tensors['token_embd.weight']
    assert shape==[2048,128000] and kind==14
    start=model.base+offset
    rows,blocks,block_bytes=128000,8,210
    row_bytes=blocks*block_bytes
    raw=np.frombuffer(model.data,dtype='u1',count=rows*row_bytes,offset=start).reshape(rows,blocks,block_bytes)
    cases=[]
    for label,folder,norm in [('Hello','trace-h9-hello','01051-result_norm.f32'),
                              ('Hello world!','trace-h9-multi','03157-result_norm.f32')]:
        folder=ROOT/'native-h9'/folder/'llama'
        norm_path=folder/norm
        x=np.fromfile(norm_path,dtype='<f4')
        logits=np.fromfile(folder/'logits.f32',dtype='<f4')
        assert x.shape==(2048,) and logits.shape==(rows,)
        d,q=q8_pack(x)
        cases.append({'prompt':label,'norm_source':str(norm_path.relative_to(ROOT)),
                      'norm_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                      'reference':logits,'best':float(logits.max()),'d':d,'q':q,
                      'prediction':np.zeros(rows,dtype='f4'),
                      'hist':{m:np.zeros(9,dtype='i8') for m in ['blockwise','suffix_l2']},
                      'lines':{m:np.zeros((start%64+rows*row_bytes+63)//64,dtype=bool)
                               for m in ['blockwise','suffix_l2']}})
    for first in range(0,rows,1024):
        coeff,dw=decode(raw[first:first+1024])
        count=len(coeff)
        norm_sq=np.sum(coeff.astype('f8')**2,axis=2)
        for case in cases:
            q,da=case['q'],case['d']
            integer=np.einsum('rbk,bk->rb',coeff.astype('i4'),q,optimize=False)
            db=np.asarray(dw*da,dtype='f4')
            partial=np.zeros((count,9),dtype='f4')
            for b in range(8):
                # F64 intermediate is a numerical screen, not a general FMA oracle.
                partial[:,b+1]=(partial[:,b].astype('f8')+
                    db[:,b].astype('f8')*integer[:,b].astype('f4').astype('f8')).astype('f4')
            case['prediction'][first:first+count]=partial[:,8]
            qnorm_sq=np.sum(q.astype('f8')**2,axis=1)
            per_block=np.abs(db.astype('f8'))*np.sqrt(norm_sq)*np.sqrt(qnorm_sq)
            bounds={'blockwise':np.cumsum(per_block[:,::-1],axis=1)[:,::-1],
                    'suffix_l2':np.sqrt(np.cumsum((dw.astype('f8')**2*norm_sq)[:,::-1],axis=1)[:,::-1])*
                    np.sqrt(np.cumsum((da.astype('f8')**2*qnorm_sq)[::-1])[::-1])[None,:]*(1+2**-22)}
            for method,bound in bounds.items():
                possible=partial[:,:8].astype('f8')+bound<case['best']-1e-3
                read=np.where(possible.any(axis=1),possible.argmax(axis=1),8)
                case['hist'][method]+=np.bincount(read,minlength=9)
                byte_start=(start%64)+(np.arange(first,first+count,dtype='i8')*row_bytes)
                line_start=byte_start//64
                line_count=np.where(read>0,(byte_start%64+read*block_bytes+63)//64,0)
                line_numbers=line_start[:,None]+np.arange(28,dtype='i8')[None,:]
                case['lines'][method][line_numbers[np.arange(28)[None,:]<line_count[:,None]]]=True
    summary=[]
    for case in cases:
        reference,prediction=case['reference'],case['prediction']
        maximum_error=float(np.max(np.abs(reference.astype('f8')-prediction.astype('f8'))))
        assert maximum_error<1e-4,(case['prompt'],maximum_error)
        assert prediction.argmax()==reference.argmax()
        for method,hist in case['hist'].items():
            skipped=int(sum((8-n)*int(c) for n,c in enumerate(hist)))
            saved=skipped*block_bytes
            lines=case['lines'][method]
            ideal_saved_cacheline_bytes=len(lines)*64-int(lines.sum())*64
            metadata=rows*8*4
            summary.append({'prompt':case['prompt'],'method':method,
                'norm_source':case['norm_source'],'norm_sha256':case['norm_sha256'],
                'known_final_best':case['best'],'top_token':int(reference.argmax()),
                'screen_max_abs_logit_error':maximum_error,
                'screen_bit_differences':int(np.count_nonzero(reference.view('u4')!=prediction.view('u4'))),
                'output_block_skip_percent':100*skipped/(rows*8),
                'output_weight_payload_bytes_avoided':saved,
                'ideal_cacheline_bytes_avoided':ideal_saved_cacheline_bytes,
                'metadata_bytes_per_token':metadata,
                'net_whole_model_traffic_saving_percent':100*(ideal_saved_cacheline_bytes-metadata)/1089828864,
                'rows_by_blocks_read':hist.tolist()})
    best_method=max(['blockwise','suffix_l2'],key=lambda m:np.mean([r['output_block_skip_percent'] for r in summary if r['method']==m]))
    best=[r for r in summary if r['method']==best_method]
    report={'scope':'Optimistic offline feasibility only; no row is skipped in the engine',
            'source_sha256':digest,'output_weight_bytes':rows*row_bytes,
            'cases':summary,'best_screened_method':best_method,
            'mean_output_blocks_skipped_percent':float(np.mean([r['output_block_skip_percent'] for r in best])),
            'mean_net_whole_model_traffic_saving_percent':float(np.mean([r['net_whole_model_traffic_saving_percent'] for r in best])),
            'limitations':['Only two saved native hidden vectors, not a diverse quality corpus.',
                'Final best score is known before scanning, unlike real best-so-far.',
                'A1e-3 guard is not a formal floating-point certificate.',
                'Current ahead-prefetch could consume skipped bytes; disabled in this ideal accounting.',
                'No norm, branch, scheduling or metadata-decode time is counted.']}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
