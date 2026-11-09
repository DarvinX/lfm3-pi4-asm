/* Numeric oracle only; never linked into the all-assembly application. */
#include "llama.h"
#include "ggml-backend.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct trace_state { const char *dir; FILE *manifest; unsigned count; };

static bool trace_tensor(struct ggml_tensor *t, bool ask, void *user) {
    struct trace_state *s = user;
    bool selected = t->type == GGML_TYPE_F32 && t->op != GGML_OP_NONE &&
                    ggml_nelements(t) <= 262144;
    if (ask) return selected;
    if (!selected) return true;
    char filename[192], path[4096], name[GGML_MAX_NAME];
    for (size_t i = 0; i < sizeof(name); ++i) {
        unsigned char c = (unsigned char)t->name[i];
        name[i] = (isalnum(c) || c == '.' || c == '_' || c == '-') ? (char)c : '_';
        if (!c) { name[i] = 0; break; }
    }
    name[sizeof(name)-1] = 0;
    snprintf(filename, sizeof(filename), "%05u-%s.f32", s->count++, name);
    snprintf(path, sizeof(path), "%s/%s", s->dir, filename);
    FILE *f = fopen(path, "wb");
    if (!f) exit(30);
    size_t bytes = ggml_nbytes(t);
    char *raw = malloc(bytes);
    if (!raw) exit(31);
    ggml_backend_tensor_get(t, raw, 0, bytes);
    for (int64_t i3=0; i3<t->ne[3]; ++i3)
    for (int64_t i2=0; i2<t->ne[2]; ++i2)
    for (int64_t i1=0; i1<t->ne[1]; ++i1)
    for (int64_t i0=0; i0<t->ne[0]; ++i0) {
        size_t at = i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];
        if (at+4 > bytes || fwrite(raw+at, 4, 1, f) != 1) exit(32);
    }
    fclose(f); free(raw);
    fprintf(s->manifest, "{\"file\":\"%s\",\"name\":\"%s\",\"op\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld]}\n",
            filename, name, ggml_op_name(t->op), (long long)t->ne[0],
            (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    fflush(s->manifest);
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2; /* model, prompt, existing output directory */
    char path[4096];
    snprintf(path, sizeof(path), "%s/manifest.jsonl", argv[3]);
    struct trace_state state = {argv[3], fopen(path,"w"), 0};
    if (!state.manifest) return 3;
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1],mp);
    if (!model) return 4;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token tokens[512];
    int n = llama_tokenize(vocab,argv[2],strlen(argv[2]),tokens,512,true,true);
    if (n<=0 || n>512) return 5;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=1;
    cp.n_threads=4; cp.n_threads_batch=4;
    cp.type_k=GGML_TYPE_F32; cp.type_v=GGML_TYPE_F32;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=false; cp.op_offload=false;
    cp.cb_eval=trace_tensor; cp.cb_eval_user_data=&state;
    struct llama_context *ctx=llama_init_from_model(model,cp);
    if (!ctx) return 6;
    struct llama_batch batch=llama_batch_get_one(tokens,n);
    if (llama_decode(ctx,batch)) return 7;
    snprintf(path,sizeof(path),"%s/logits.f32",argv[3]);
    FILE *f=fopen(path,"wb");
    if (!f) return 8;
    int nv=llama_vocab_n_tokens(vocab),best=0;
    float *logits=llama_get_logits_ith(ctx,-1);
    fwrite(logits,sizeof(float),nv,f); fclose(f);
    for (int i=1;i<nv;++i) if (logits[i]>logits[best]) best=i;
    printf("prompt_tokens=%d greedy_token=%d trace_tensors=%u\n",n,best,state.count);
    fclose(state.manifest); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
