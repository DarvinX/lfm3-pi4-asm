/* Matched streaming/latency adapter; pinned llama.cpp core is unchanged.
 * Original bench_llama.c and parity-llama binary remain immutable controls.
 */
#define _POSIX_C_SOURCE 200809L
#include "llama.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
static double now(void){struct timespec ts;if(clock_gettime(CLOCK_MONOTONIC,&ts))exit(10);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int greedy(const float *logits,int n){int best=0;for(int i=0;i<n;++i){if(!isfinite(logits[i]))exit(11);if(logits[i]>logits[best])best=i;}return best;}
int main(int argc,char **argv){
    if(argc!=5)return 2;
    double request_start=now();
    int maximum=atoi(argv[3]),threads=atoi(argv[4]);
    if(maximum<1||maximum>128||threads<1||threads>4)return 2;
    const char *dt=getenv("LFM_DECODE_THREADS");int decode_threads=dt?atoi(dt):threads;
    if(decode_threads<1||decode_threads>threads)return 2;
    double load_start=now();llama_backend_init();
    struct llama_model_params mp=llama_model_default_params();mp.n_gpu_layers=0;
    struct llama_model *m=llama_model_load_from_file(argv[1],mp);if(!m)return 3;
    const struct llama_vocab *v=llama_model_get_vocab(m);llama_token tokens[512];
    int n=llama_tokenize(v,argv[2],(int)strlen(argv[2]),tokens,512,true,true);
    if(n<=0||n+maximum>512)return 4;
    struct llama_context_params cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=512;cp.n_ubatch=1;cp.n_threads=threads;cp.n_threads_batch=threads;
    cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=false;cp.op_offload=false;
    struct llama_context *ctx=llama_init_from_model(m,cp);if(!ctx)return 5;
    int prefill_threads=llama_n_threads_batch(ctx);if(prefill_threads!=threads||llama_n_threads(ctx)!=threads)return 14;
    double load_end=now();struct llama_batch prompt=llama_batch_get_one(tokens,n);
    if(llama_decode(ctx,prompt))return 6;
    double prefill_end=now();llama_set_n_threads(ctx,decode_threads,threads);
    if(llama_n_threads(ctx)!=decode_threads||llama_n_threads_batch(ctx)!=threads)return 14;
    int generated=0,evaluations=0,first_bytes=0;double first_output=-1;
    FILE *token_ids=NULL;const char *token_path=getenv("LFM_TOKEN_IDS");
    if(token_path&&!(token_ids=fopen(token_path,"wb")))return 12;
    int nv=llama_vocab_n_tokens(v);const int ignore_eos=getenv("LFM_BENCH_IGNORE_EOS")!=NULL;
    for(int i=0;i<maximum;++i){
        llama_token token=greedy(llama_get_logits_ith(ctx,-1),nv);
        if(!ignore_eos&&token==llama_vocab_eos(v))break;
        if(token_ids&&fwrite(&token,sizeof(token),1,token_ids)!=1)return 13;
        char text[16384];int size=llama_token_to_piece(v,token,text,(int)sizeof(text),0,true);
        if(size<0||fwrite(text,1,size,stdout)!=(size_t)size||fflush(stdout))return 7;
        if(size>0&&first_output<0){first_output=now();first_bytes=size;}
        ++generated;if(i+1==maximum)break;
        struct llama_batch one=llama_batch_get_one(&token,1);if(llama_decode(ctx,one))return 8;++evaluations;
    }
    if(fflush(stdout))return 7;
    if(token_ids)fclose(token_ids);
    double decode_end=now();struct rusage usage;if(getrusage(RUSAGE_SELF,&usage))return 9;
    fprintf(stderr,"prefill_tokens=%d generated_tokens=%d decode_steps=%d peak_rss_kib=%ld load_seconds=%.9f prefill_seconds=%.9f decode_seconds=%.9f\n",n,generated,evaluations,usage.ru_maxrss,load_end-load_start,prefill_end-load_end,decode_end-prefill_end);
    if(first_output>=0)fprintf(stderr,"LATENCY ttft_seconds=%.9f request_seconds=%.9f prefill_ttft_seconds=%.9f first_piece_bytes=%d prefill_threads=%d decode_threads=%d\n",first_output-request_start,decode_end-request_start,first_output-load_end,first_bytes,prefill_threads,decode_threads);
    llama_free(ctx);llama_model_free(m);llama_backend_free();return 0;
}
