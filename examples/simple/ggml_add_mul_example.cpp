#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(void) {
    ggml_time_init();

    int n_threads = 4;
    if (const char * env_threads = getenv("GGML_EXAMPLE_THREADS")) {
        int parsed = atoi(env_threads);
        if (parsed > 0) n_threads = parsed;
    }

    int64_t n_elems = 2048;
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    if (tp == NULL) {
        fprintf(stderr, "failed to create ggml threadpool\n");
        return 1;
    }

    size_t ctx_size = 0;
    const size_t tensor_bytes = (size_t) n_elems * ggml_type_size(GGML_TYPE_F32);

    ctx_size += 4 * tensor_bytes;               // a, b, sum, out data
    ctx_size += 4 * ggml_tensor_overhead();     // a, b, sum, out tensor metadata
    ctx_size += ggml_graph_overhead();          // graph
    ctx_size += 1024;                           // extra slack

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "failed to init ggml context\n");
        ggml_threadpool_free(tp);
        return 1;
    }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elems);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elems);

    std::vector<float> a_data((size_t) n_elems);
    std::vector<float> b_data((size_t) n_elems);

    for (int64_t i = 0; i < n_elems; ++i) {
        a_data[(size_t) i] = (float) (i + 1);  // 1, 2, 3, ...
        b_data[(size_t) i] = (float) (i + 5);  // 5, 6, 7, ...
    }

    memcpy(a->data, a_data.data(), (size_t) n_elems * sizeof(float));
    memcpy(b->data, b_data.data(), (size_t) n_elems * sizeof(float));

    struct ggml_tensor * sum = ggml_add(ctx, a, b);
    struct ggml_tensor * out = ggml_mul(ctx, sum, sum);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *) malloc(plan.work_size);
        if (plan.work_data == NULL) {
            fprintf(stderr, "failed to allocate work buffer\n");
            ggml_free(ctx);
            ggml_threadpool_free(tp);
            return 1;
        }
    } else {
        plan.work_data = NULL;
    }

    ggml_graph_compute(gf, &plan);
    free(plan.work_data);

    float * out_data = (float *) out->data;
    printf("result:");
    for (int64_t i = 0; i < n_elems; i++) {
        printf(" %.2f", out_data[i]);
    }
    printf("\n");

    ggml_free(ctx);
    ggml_threadpool_free(tp);
    return 0;
}
