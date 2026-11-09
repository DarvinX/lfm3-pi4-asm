#!/usr/bin/env python3
"""Prepare H34 centered4-bit auxiliary head and certified norm ceilings.

Original GGUF is unchanged. These data do not form an inference engine and
cannot select outputs without original-score refinement and bound checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
from prepare import Reader,EXPECTED
from coarse_vocab_feasibility import codes


def pack(raw):
    q,scale,_=codes(raw)
    c=((q+32)//4).astype('u1')
    out=np.empty((*raw.shape[:2],146),dtype='u1')
    reconstructed=np.empty_like(c)
    for pair in range(4):
        payload=c[:,:,pair*64:pair*64+32]|(c[:,:,pair*64+32:pair*64+64]<<4)
        out[:,:,pair*32:pair*32+32]=payload
        reconstructed[:,:,pair*64:pair*64+32]=payload&15
        reconstructed[:,:,pair*64+32:pair*64+64]=payload>>4
    assert np.array_equal(c,reconstructed),'compact nibble address roundtrip'
    out[:,:,128:144]=raw[:,:,192:208]
    out[:,:,144:146]=raw[:,:,208:210]
    coarse=reconstructed.astype('i2')*4-30
    residual=(q-coarse)*scale
    squared=np.sum(residual.astype('i8')**2,axis=2)
    assert np.all((squared>=0)&(squared<=2**24))
    root=np.sqrt(squared.astype('f8')).astype('f4')
    norms=np.where(squared==0,np.float32(0),np.nextafter(root,np.float32(np.inf))).astype('<f4')
    # FP32*FP32 squares are exact in FP64 in this range; no sqrt assumption.
    assert np.all(norms.astype('f8')**2>=squared.astype('f8'))
    assert np.all(np.isfinite(norms)&((norms==0)|(norms>=np.finfo('f4').tiny)))
    return out,norms


def fileinfo(path,crc):
    return {'file':path.name,'bytes':path.stat().st_size,
            'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'crc32':crc}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--out',type=Path,default=Path('build/h34-coarse-v1'))
    a=p.parse_args()
    if a.out.exists():p.error('fresh output directory required')
    model=Reader(a.model);digest=hashlib.sha256(model.data).hexdigest();assert digest==EXPECTED
    offset,shape,kind=model.tensors['token_embd.weight'];assert shape==[2048,128000] and kind==14
    raw=np.frombuffer(model.data,dtype='u1',count=128000*8*210,offset=model.base+offset).reshape(128000,8,210)
    a.out.mkdir(parents=True)
    data_path=a.out/'coarse-output.bin';norm_path=a.out/'residual-norms.f32'
    data_pending=data_path.with_suffix('.pending');norm_pending=norm_path.with_suffix('.pending')
    data_crc=norm_crc=0
    with data_pending.open('xb') as data,norm_pending.open('xb') as norms:
        for first in range(0,128000,1024):
            packed,ceilings=pack(raw[first:first+1024]);payload=packed.tobytes();bounds=ceilings.tobytes()
            data.write(payload);norms.write(bounds)
            data_crc=zlib.crc32(payload,data_crc);norm_crc=zlib.crc32(bounds,norm_crc)
            if first==0:(a.out/'fixture-head-2048.raw').write_bytes(packed[:16].tobytes())
    data_pending.replace(data_path);norm_pending.replace(norm_path)
    assert data_path.stat().st_size==149504000 and norm_path.stat().st_size==4096000
    off,shape,kind=model.tensors['blk.0.ffn_down.weight'];assert shape==[7680,2048] and kind==14
    fixture=np.frombuffer(model.data,dtype='u1',count=16*30*210,offset=model.base+off).reshape(16,30,210)
    packed,_=pack(fixture);(a.out/'fixture-ffdown-7680.raw').write_bytes(packed.tobytes())
    info={'format':'h34-centered-q4-v1','scope':'auxiliary compact head and norm data only; no runtime/controller yet',
          'source_sha256':digest,'rows':128000,'columns':2048,'blocks_per_row':8,'block_bytes':146,
          'coarse_code':'4*c-30; c=floor((original_signed_Q6+32)/4)',
          'code_values_roundtrip_checked':128000*2048,'norm_ceilings_square_verified':128000*8,
          'original_signed_scales_and_half_d_preserved':True,'norm_zero_stored_exactly_zero':True,
          'coarse':fileinfo(data_path,data_crc),'norms':fileinfo(norm_path,norm_crc),
          'fixtures':[{'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(a.out.glob('fixture-*.raw'))]}
    (a.out/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    (a.out/'coarse-index.inc').write_text('\n'.join([
        '/* Auxiliary H34 data bound to the compiled exact checkpoint. */',
        '.equ H34_ROWS,128000','.equ H34_COLUMNS,2048','.equ H34_BLOCK_BYTES,146',
        '.equ H34_COARSE_BYTES,149504000','.equ H34_NORM_BYTES,4096000',
        f'.equ H34_COARSE_CRC,{data_crc}',f'.equ H34_NORM_CRC,{norm_crc}','']))
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
