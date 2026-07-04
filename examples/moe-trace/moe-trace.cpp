#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "moevm_runtime.hpp"

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum class moevm_copy_mode {
    async,
    memcpy,
};

struct moevm_local_params {
    std::string index_path;
    std::string component_index_path;
    std::string sidecar_path;
    double arena_gib = 8.0;
    double inflight_mib = 64.0;
    int threads = 4;
    int prefetch_topk = 10;
    int copy_debug = 0;
    bool direct_mmap = false;
    bool nocache = true;
    bool shadow_ids = false;
    int compact_layer = -1;
    int compact_k_max = 64;
    int pool_layer = -1;
    int pool_capacity = 10;
    bool resident_cache = false;
    bool defer_input_wait = true;
    bool self_test_pool_pinning = false;
    bool test_force_pinned_fence = false;
    bool speculative = false;
    uint32_t inject_delay_max_us = 0;
    uint64_t inject_delay_seed = 0x9e3779b97f4a7c15ull;
    moevm_copy_mode copy_mode = moevm_copy_mode::async;
};

struct copy_prefetch_cache_entry {
    int64_t decode_base = -1;
    std::vector<uint32_t> ids;
};

enum class shadow_ids_kind {
    verbatim,
    compact,
    pool,
};

struct shadow_ids_slot {
    shadow_ids_kind kind = shadow_ids_kind::verbatim;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
    enum ggml_type type = GGML_TYPE_I32;
    int n_dims = 0;
    int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
    size_t bytes = 0;
};

struct compact_weight_slot {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t capacity = 0;
    size_t expert_size = 0;
    moevm::ExpertComponent component = moevm::ExpertComponent::Gate;
};

struct pool_weight_slot {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
    enum ggml_type type = GGML_TYPE_COUNT;
    uint32_t layer = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t capacity = 0;
    size_t expert_size = 0;
    moevm::ExpertComponent component = moevm::ExpertComponent::Gate;
    std::vector<int32_t> loaded_expert_by_slot;
};

struct layer_pool_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    uint32_t layer = 0;
    int64_t n_expert = 0;
    int capacity = 0;
    uint64_t tick = 0;
    std::vector<int32_t> slot_by_expert;
    std::vector<int32_t> expert_by_slot;
    std::vector<uint64_t> last_used_by_slot;
    std::vector<uint64_t> last_epoch_by_slot;
};

struct pool_slot_plan {
    std::vector<int32_t> slot_by_selected;
    std::vector<int32_t> slot_by_expert;
    std::vector<int32_t> expert_by_slot;
    uint64_t evictions = 0;
};

struct resident_weight_cache_slot {
    ggml_tensor * dst_tensor = nullptr;
    ggml_backend_buffer_t dst_buffer = nullptr;
    void * dst_data = nullptr;
    const ggml_tensor * src_tensor = nullptr;
    uint32_t layer = 0;
    int64_t n_expert = 0;
    size_t expert_size = 0;
    moevm::ExpertComponent component = moevm::ExpertComponent::Gate;
    std::vector<uint8_t> loaded_by_expert;
};

struct moe_trace_state {
    FILE * out = nullptr;
    FILE * logits_out = nullptr;
    std::vector<uint8_t> scratch;
    std::vector<llama_token> generated_tokens;
    std::vector<copy_prefetch_cache_entry> copy_prefetch_cache;
    std::vector<shadow_ids_slot> shadow_ids_slots;
    std::vector<compact_weight_slot> compact_weight_slots;
    std::vector<pool_weight_slot> pool_weight_slots;
    std::vector<layer_pool_state> layer_pools;
    std::vector<resident_weight_cache_slot> resident_weight_cache_slots;
    std::unique_ptr<moevm::Runtime> runtime;
    std::mutex runtime_error_mu;
    std::string runtime_error;
    std::string runtime_error_last;
    int candidate_limit = 0;
    int moevm_prefetch_topk = 10;
    uint32_t logits_n_vocab = 0;
    uint32_t logits_rows = 0;
    int64_t decode_base = 0;
    int64_t chunk_base = 0;
    int64_t last_chunk_tokens = 0;
    int last_layer = -1;
    int64_t records = 0;
    uint64_t moevm_prefetch_records = 0;
    uint64_t moevm_copy_prefetches = 0;
    uint64_t moevm_copy_prefetch_skips = 0;
    uint64_t moevm_copy_hook_calls = 0;
    uint64_t moevm_copy_hook_handled = 0;
    uint64_t moevm_copy_experts = 0;
    uint64_t moevm_copy_bytes = 0;
    uint64_t moevm_copy_async_ops = 0;
    uint64_t moevm_copy_memcpy_ops = 0;
    uint64_t moevm_shadow_ids_attempts = 0;
    uint64_t moevm_shadow_ids_applied = 0;
    uint64_t moevm_shadow_ids_fallback = 0;
    uint64_t moevm_compact_attempts = 0;
    uint64_t moevm_compact_applied = 0;
    uint64_t moevm_compact_fallback = 0;
    uint64_t moevm_fallback_k_overflow = 0;
    uint64_t moevm_pool_attempts = 0;
    uint64_t moevm_pool_applied = 0;
    uint64_t moevm_pool_fallback = 0;
    uint64_t moevm_pool_component_hits = 0;
    uint64_t moevm_pool_component_misses = 0;
    uint64_t moevm_pool_evictions = 0;
    uint64_t moevm_pool_fallback_shape = 0;
    uint64_t moevm_pool_fallback_ids_read = 0;
    uint64_t moevm_pool_fallback_alloc = 0;
    uint64_t moevm_pool_fallback_alloc_layer = 0;
    uint64_t moevm_pool_fallback_alloc_pool = 0;
    uint64_t moevm_pool_fallback_alloc_pool_data = 0;
    uint64_t moevm_pool_fallback_alloc_shadow = 0;
    uint64_t moevm_pool_fallback_alloc_shadow_data = 0;
    uint64_t moevm_pool_fallback_alloc_shadow_bytes = 0;
    uint64_t moevm_pool_fallback_plan = 0;
    uint64_t moevm_pool_fallback_remap = 0;
    uint64_t moevm_pool_fallback_fill = 0;
    uint64_t moevm_pool_submit_epoch = 0;
    uint64_t moevm_pool_completed_epoch = 0;
    uint64_t moevm_pool_epoch_drains = 0;
    uint64_t moevm_pool_forced_fences = 0;
    uint64_t moevm_pool_pinned_eviction_skips = 0;
    uint64_t moevm_delay_injections = 0;
    uint64_t moevm_delay_us = 0;
    uint64_t moevm_delay_state = 0x9e3779b97f4a7c15ull;
    uint64_t moevm_resident_cache_hits = 0;
    uint64_t moevm_resident_cache_misses = 0;
    uint64_t moevm_resident_cache_resets = 0;
    uint64_t moevm_runtime_errors = 0;
    bool moevm_shadow_ids = false;
    int moevm_compact_layer = -1;
    int moevm_compact_k_max = 64;
    int moevm_pool_layer = -1;
    int moevm_pool_capacity = 10;
    bool moevm_resident_cache = false;
    bool moevm_defer_input_wait = false;
    bool moevm_test_force_pinned_fence = false;
    bool moevm_test_force_pinned_fence_done = false;
    uint32_t moevm_inject_delay_max_us = 0;
    moevm_copy_mode moevm_copy_mode_value = moevm_copy_mode::async;

    ~moe_trace_state() {
        close_logits_out();
        clear_shadow_ids();
    }

    bool open_logits_out(const std::string & path, uint32_t n_vocab) {
        logits_out = std::fopen(path.c_str(), "wb");
        if (!logits_out) {
            return false;
        }
        logits_n_vocab = n_vocab;
        logits_rows = 0;

        const char magic[8] = {'M', 'O', 'E', 'V', 'M', 'L', 'G', '1'};
        const uint32_t version = 1;
        const uint32_t rows = 0;
        const uint32_t reserved = 0;
        return std::fwrite(magic, 1, sizeof(magic), logits_out) == sizeof(magic) &&
            std::fwrite(&version, sizeof(version), 1, logits_out) == 1 &&
            std::fwrite(&logits_n_vocab, sizeof(logits_n_vocab), 1, logits_out) == 1 &&
            std::fwrite(&rows, sizeof(rows), 1, logits_out) == 1 &&
            std::fwrite(&reserved, sizeof(reserved), 1, logits_out) == 1;
    }

    bool write_logits_row(uint32_t step, int32_t sampled_token, const float * logits) {
        if (!logits_out) {
            return true;
        }
        if (!logits) {
            runtime_error = "missing logits pointer";
            return false;
        }
        if (std::fwrite(&step, sizeof(step), 1, logits_out) != 1 ||
                std::fwrite(&sampled_token, sizeof(sampled_token), 1, logits_out) != 1 ||
                std::fwrite(logits, sizeof(float), logits_n_vocab, logits_out) != logits_n_vocab) {
            runtime_error = "failed writing logits row";
            return false;
        }
        ++logits_rows;
        return true;
    }

    void close_logits_out() {
        if (!logits_out) {
            return;
        }
        std::fflush(logits_out);
        if (std::fseek(logits_out, 16, SEEK_SET) == 0) {
            std::fwrite(&logits_rows, sizeof(logits_rows), 1, logits_out);
        }
        std::fclose(logits_out);
        logits_out = nullptr;
    }

    void clear_shadow_ids() {
        for (shadow_ids_slot & slot : shadow_ids_slots) {
            if (slot.buffer) {
                ggml_backend_buffer_free(slot.buffer);
                slot.buffer = nullptr;
            }
            if (slot.ctx) {
                ggml_free(slot.ctx);
                slot.ctx = nullptr;
            }
            slot.tensor = nullptr;
        }
        shadow_ids_slots.clear();
        for (compact_weight_slot & slot : compact_weight_slots) {
            if (slot.buffer) {
                ggml_backend_buffer_free(slot.buffer);
                slot.buffer = nullptr;
            }
            if (slot.ctx) {
                ggml_free(slot.ctx);
                slot.ctx = nullptr;
            }
            slot.tensor = nullptr;
        }
        compact_weight_slots.clear();
        for (pool_weight_slot & slot : pool_weight_slots) {
            if (slot.buffer) {
                ggml_backend_buffer_free(slot.buffer);
                slot.buffer = nullptr;
            }
            if (slot.ctx) {
                ggml_free(slot.ctx);
                slot.ctx = nullptr;
            }
            slot.tensor = nullptr;
        }
        pool_weight_slots.clear();
        layer_pools.clear();
        resident_weight_cache_slots.clear();
    }
};

static void moevm_set_runtime_error(moe_trace_state * state, const std::string & message) {
    if (!state) {
        return;
    }

    bool log_first = false;
    {
        std::lock_guard<std::mutex> lock(state->runtime_error_mu);
        ++state->moevm_runtime_errors;
        state->runtime_error_last = message;
        if (state->runtime_error.empty()) {
            state->runtime_error = message;
            log_first = true;
        }
    }

    if (log_first) {
        LOG_ERR("MoEVM runtime error: %s\n", message.c_str());
    }
}

struct moevm_record_guard {
    moevm::Runtime * runtime = nullptr;
    moevm::RecordView record{};

    moevm_record_guard(moevm::Runtime * runtime_in, moevm::RecordView record_in)
        : runtime(runtime_in), record(record_in) {}

    moevm_record_guard(const moevm_record_guard &) = delete;
    moevm_record_guard & operator=(const moevm_record_guard &) = delete;

    ~moevm_record_guard() {
        if (runtime != nullptr && record.data != nullptr) {
            runtime->release_record(record);
        }
    }
};

static const char * moevm_copy_mode_name(moevm_copy_mode mode) {
    switch (mode) {
        case moevm_copy_mode::async:
            return "async";
        case moevm_copy_mode::memcpy:
            return "memcpy";
    }
    return "unknown";
}

static bool parse_moevm_copy_mode(const std::string & value, moevm_copy_mode & mode) {
    if (value == "async") {
        mode = moevm_copy_mode::async;
        return true;
    }
    if (value == "memcpy") {
        mode = moevm_copy_mode::memcpy;
        return true;
    }
    return false;
}

static bool parse_local_arg(
        int argc,
        char ** argv,
        std::vector<char *> & filtered,
        std::string & trace_out,
        std::string & logits_out,
        std::string & generated_out,
        int & candidate_limit,
        moevm_local_params & moevm_params) {
    filtered.clear();
    filtered.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trace-out") {
            if (i + 1 >= argc) {
                LOG_ERR("--trace-out requires a path\n");
                return false;
            }
            trace_out = argv[++i];
            continue;
        }
        const std::string prefix = "--trace-out=";
        if (arg.rfind(prefix, 0) == 0) {
            trace_out = arg.substr(prefix.size());
            continue;
        }
        if (arg == "--logits-out") {
            if (i + 1 >= argc) {
                LOG_ERR("--logits-out requires a path\n");
                return false;
            }
            logits_out = argv[++i];
            continue;
        }
        const std::string logits_prefix = "--logits-out=";
        if (arg.rfind(logits_prefix, 0) == 0) {
            logits_out = arg.substr(logits_prefix.size());
            continue;
        }
        if (arg == "--generated-out") {
            if (i + 1 >= argc) {
                LOG_ERR("--generated-out requires a path\n");
                return false;
            }
            generated_out = argv[++i];
            continue;
        }
        const std::string generated_prefix = "--generated-out=";
        if (arg.rfind(generated_prefix, 0) == 0) {
            generated_out = arg.substr(generated_prefix.size());
            continue;
        }
        if (arg == "--trace-candidates") {
            if (i + 1 >= argc) {
                LOG_ERR("--trace-candidates requires an integer\n");
                return false;
            }
            candidate_limit = std::atoi(argv[++i]);
            continue;
        }
        const std::string cand_prefix = "--trace-candidates=";
        if (arg.rfind(cand_prefix, 0) == 0) {
            candidate_limit = std::atoi(arg.substr(cand_prefix.size()).c_str());
            continue;
        }
        if (arg == "--moevm-index") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-index requires a path\n");
                return false;
            }
            moevm_params.index_path = argv[++i];
            continue;
        }
        if (arg.rfind("--moevm-index=", 0) == 0) {
            moevm_params.index_path = arg.substr(std::strlen("--moevm-index="));
            continue;
        }
        if (arg == "--moevm-component-index") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-component-index requires a path\n");
                return false;
            }
            moevm_params.component_index_path = argv[++i];
            continue;
        }
        if (arg.rfind("--moevm-component-index=", 0) == 0) {
            moevm_params.component_index_path = arg.substr(std::strlen("--moevm-component-index="));
            continue;
        }
        if (arg == "--moevm-sidecar") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-sidecar requires a path\n");
                return false;
            }
            moevm_params.sidecar_path = argv[++i];
            continue;
        }
        if (arg.rfind("--moevm-sidecar=", 0) == 0) {
            moevm_params.sidecar_path = arg.substr(std::strlen("--moevm-sidecar="));
            continue;
        }
        if (arg == "--moevm-arena-gib") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-arena-gib requires a number\n");
                return false;
            }
            moevm_params.arena_gib = std::atof(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-arena-gib=", 0) == 0) {
            moevm_params.arena_gib = std::atof(arg.substr(std::strlen("--moevm-arena-gib=")).c_str());
            continue;
        }
        if (arg == "--moevm-inflight-mib") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-inflight-mib requires a number\n");
                return false;
            }
            moevm_params.inflight_mib = std::atof(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-inflight-mib=", 0) == 0) {
            moevm_params.inflight_mib = std::atof(arg.substr(std::strlen("--moevm-inflight-mib=")).c_str());
            continue;
        }
        if (arg == "--moevm-threads") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-threads requires an integer\n");
                return false;
            }
            moevm_params.threads = std::atoi(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-threads=", 0) == 0) {
            moevm_params.threads = std::atoi(arg.substr(std::strlen("--moevm-threads=")).c_str());
            continue;
        }
        if (arg == "--moevm-prefetch-topk") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-prefetch-topk requires an integer\n");
                return false;
            }
            moevm_params.prefetch_topk = std::atoi(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-prefetch-topk=", 0) == 0) {
            moevm_params.prefetch_topk = std::atoi(arg.substr(std::strlen("--moevm-prefetch-topk=")).c_str());
            continue;
        }
        if (arg == "--moevm-copy-debug") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-copy-debug requires an integer\n");
                return false;
            }
            moevm_params.copy_debug = std::atoi(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-copy-debug=", 0) == 0) {
            moevm_params.copy_debug = std::atoi(arg.substr(std::strlen("--moevm-copy-debug=")).c_str());
            continue;
        }
        if (arg == "--moevm-copy-mode") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-copy-mode requires async or memcpy\n");
                return false;
            }
            if (!parse_moevm_copy_mode(argv[++i], moevm_params.copy_mode)) {
                LOG_ERR("--moevm-copy-mode must be async or memcpy\n");
                return false;
            }
            continue;
        }
        if (arg.rfind("--moevm-copy-mode=", 0) == 0) {
            if (!parse_moevm_copy_mode(arg.substr(std::strlen("--moevm-copy-mode=")), moevm_params.copy_mode)) {
                LOG_ERR("--moevm-copy-mode must be async or memcpy\n");
                return false;
            }
            continue;
        }
        if (arg == "--moevm-direct-mmap") {
            moevm_params.direct_mmap = true;
            continue;
        }
        if (arg == "--moevm-cache") {
            moevm_params.nocache = false;
            continue;
        }
        if (arg == "--moevm-nocache") {
            moevm_params.nocache = true;
            continue;
        }
        if (arg == "--moevm-shadow-ids") {
            moevm_params.shadow_ids = true;
            continue;
        }
        if (arg.rfind("--moevm-shadow-ids=", 0) == 0) {
            moevm_params.shadow_ids = std::atoi(arg.substr(std::strlen("--moevm-shadow-ids=")).c_str()) != 0;
            continue;
        }
        if (arg == "--moevm-compact-layer") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-compact-layer requires an integer layer, all, or -1 to disable\n");
                return false;
            }
            const std::string value = argv[++i];
            moevm_params.compact_layer = value == "all" ? -2 : std::atoi(value.c_str());
            continue;
        }
        if (arg.rfind("--moevm-compact-layer=", 0) == 0) {
            const std::string value = arg.substr(std::strlen("--moevm-compact-layer="));
            moevm_params.compact_layer = value == "all" ? -2 : std::atoi(value.c_str());
            continue;
        }
        if (arg == "--moevm-compact-k-max") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-compact-k-max requires an integer\n");
                return false;
            }
            moevm_params.compact_k_max = std::atoi(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-compact-k-max=", 0) == 0) {
            moevm_params.compact_k_max = std::atoi(arg.substr(std::strlen("--moevm-compact-k-max=")).c_str());
            continue;
        }
        if (arg == "--moevm-pool-layer") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-pool-layer requires an integer layer, all, or -1 to disable\n");
                return false;
            }
            const std::string value = argv[++i];
            moevm_params.pool_layer = value == "all" ? -2 : std::atoi(value.c_str());
            continue;
        }
        if (arg.rfind("--moevm-pool-layer=", 0) == 0) {
            const std::string value = arg.substr(std::strlen("--moevm-pool-layer="));
            moevm_params.pool_layer = value == "all" ? -2 : std::atoi(value.c_str());
            continue;
        }
        if (arg == "--moevm-pool-capacity") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-pool-capacity requires an integer\n");
                return false;
            }
            moevm_params.pool_capacity = std::atoi(argv[++i]);
            continue;
        }
        if (arg.rfind("--moevm-pool-capacity=", 0) == 0) {
            moevm_params.pool_capacity = std::atoi(arg.substr(std::strlen("--moevm-pool-capacity=")).c_str());
            continue;
        }
        if (arg == "--moevm-defer-input-wait") {
            moevm_params.defer_input_wait = true;
            continue;
        }
        if (arg == "--moevm-no-defer-input-wait") {
            moevm_params.defer_input_wait = false;
            continue;
        }
        if (arg.rfind("--moevm-defer-input-wait=", 0) == 0) {
            moevm_params.defer_input_wait = std::atoi(arg.substr(std::strlen("--moevm-defer-input-wait=")).c_str()) != 0;
            continue;
        }
        if (arg == "--moevm-self-test-pool-pinning") {
            moevm_params.self_test_pool_pinning = true;
            continue;
        }
        if (arg == "--moevm-test-force-pinned-fence") {
            moevm_params.test_force_pinned_fence = true;
            continue;
        }
        if (arg.rfind("--moevm-test-force-pinned-fence=", 0) == 0) {
            moevm_params.test_force_pinned_fence = std::atoi(arg.substr(std::strlen("--moevm-test-force-pinned-fence=")).c_str()) != 0;
            continue;
        }
        if (arg == "--moevm-speculative") {
            moevm_params.speculative = true;
            continue;
        }
        if (arg.rfind("--moevm-speculative=", 0) == 0) {
            moevm_params.speculative = std::atoi(arg.substr(std::strlen("--moevm-speculative=")).c_str()) != 0;
            continue;
        }
        if (arg == "--moevm-inject-delay-us") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-inject-delay-us requires an integer\n");
                return false;
            }
            moevm_params.inject_delay_max_us = (uint32_t) std::max(0, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--moevm-inject-delay-us=", 0) == 0) {
            moevm_params.inject_delay_max_us = (uint32_t) std::max(0, std::atoi(arg.substr(std::strlen("--moevm-inject-delay-us=")).c_str()));
            continue;
        }
        if (arg == "--moevm-inject-delay-seed") {
            if (i + 1 >= argc) {
                LOG_ERR("--moevm-inject-delay-seed requires an integer\n");
                return false;
            }
            moevm_params.inject_delay_seed = std::strtoull(argv[++i], nullptr, 0);
            continue;
        }
        if (arg.rfind("--moevm-inject-delay-seed=", 0) == 0) {
            moevm_params.inject_delay_seed = std::strtoull(arg.substr(std::strlen("--moevm-inject-delay-seed=")).c_str(), nullptr, 0);
            continue;
        }
        if (arg == "--moevm-resident-cache") {
            LOG_ERR("--moevm-resident-cache is disabled after failed parity; use --moevm-resident-cache-unsafe only for negative R&D reproduction\n");
            return false;
        }
        if (arg == "--moevm-resident-cache-unsafe") {
            moevm_params.resident_cache = true;
            continue;
        }
        if (arg.rfind("--moevm-resident-cache-unsafe=", 0) == 0) {
            moevm_params.resident_cache = std::atoi(arg.substr(std::strlen("--moevm-resident-cache-unsafe=")).c_str()) != 0;
            continue;
        }
        filtered.push_back(argv[i]);
    }

    return true;
}

enum class trace_tensor_kind {
    none,
    topk,
    candidates,
};

static bool parse_layer_with_prefix(const char * name, const char * prefix, int & layer) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(name + n, &end, 10);
    if (errno || end == name + n || *end != '\0' || value < 0) {
        return false;
    }
    layer = (int) value;
    return true;
}

static trace_tensor_kind parse_layer(const char * name, int & layer) {
    if (parse_layer_with_prefix(name, "ffn_moe_topk-", layer)) {
        return trace_tensor_kind::topk;
    }
    if (parse_layer_with_prefix(name, "ffn_moe_argsort-", layer)) {
        return trace_tensor_kind::candidates;
    }
    return trace_tensor_kind::none;
}

static int64_t get_i64(const uint8_t * data, const ggml_tensor * t, int64_t i0, int64_t i1) {
    const size_t off = i0 * t->nb[0] + i1 * t->nb[1];
    switch (t->type) {
        case GGML_TYPE_I8:
            return *(const int8_t *) (data + off);
        case GGML_TYPE_I16:
            return *(const int16_t *) (data + off);
        case GGML_TYPE_I32:
            return *(const int32_t *) (data + off);
        case GGML_TYPE_I64:
            return *(const int64_t *) (data + off);
        case GGML_TYPE_F32:
            return (int64_t) *(const float *) (data + off);
        default:
            return -1;
    }
}

static bool moevm_tensor_i32_index(const ggml_tensor * tensor, int64_t i0, int64_t i1, size_t & index) {
    if (!tensor || tensor->type != GGML_TYPE_I32 || i0 < 0 || i1 < 0 || i0 >= tensor->ne[0] || i1 >= tensor->ne[1]) {
        return false;
    }
    const uint64_t off = (uint64_t) i0 * tensor->nb[0] + (uint64_t) i1 * tensor->nb[1];
    if (off % sizeof(int32_t) != 0) {
        return false;
    }
    index = (size_t) (off / sizeof(int32_t));
    return true;
}

static bool build_shadow_ids_remap_i32(
        const ggml_tensor * ids_tensor,
        const int32_t * ids_data,
        size_t ids_data_count,
        ggml_tensor * shadow,
        const std::vector<int32_t> & remap,
        int64_t n_expert,
        std::vector<int32_t> & shadow_ids_tmp) {
    if (!ids_tensor || !ids_data || !shadow || !shadow->data ||
            ids_tensor->type != GGML_TYPE_I32 || shadow->type != GGML_TYPE_I32 ||
            ids_tensor->ne[0] != shadow->ne[0] || ids_tensor->ne[1] != shadow->ne[1]) {
        return false;
    }
    if (n_expert <= 0 || (size_t) n_expert > remap.size()) {
        return false;
    }

    const size_t shadow_count = ggml_nbytes(shadow) / sizeof(int32_t);
    shadow_ids_tmp.assign(shadow_count, 0);

    for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; ++i0) {
            size_t src_index = 0;
            size_t dst_index = 0;
            if (!moevm_tensor_i32_index(ids_tensor, i0, i1, src_index) ||
                    !moevm_tensor_i32_index(shadow, i0, i1, dst_index) ||
                    src_index >= ids_data_count || dst_index >= shadow_count) {
                return false;
            }
            const int32_t expert = ids_data[src_index];
            if (expert < 0 || expert >= n_expert || remap[(size_t) expert] < 0) {
                return false;
            }
            shadow_ids_tmp[dst_index] = remap[(size_t) expert];
        }
    }

    return true;
}

static bool parse_routed_tensor_component(const char * name, uint32_t & layer, moevm::ExpertComponent & component) {
    constexpr const char * prefix = "blk.";
    if (std::strncmp(name, prefix, std::strlen(prefix)) != 0) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long layer_value = std::strtol(name + std::strlen(prefix), &end, 10);
    if (errno || end == name + std::strlen(prefix) || layer_value < 0) {
        return false;
    }
    layer = (uint32_t) layer_value;

    if (std::strcmp(end, ".ffn_gate_exps.weight") == 0) {
        component = moevm::ExpertComponent::Gate;
        return true;
    }
    if (std::strcmp(end, ".ffn_up_exps.weight") == 0) {
        component = moevm::ExpertComponent::Up;
        return true;
    }
    if (std::strcmp(end, ".ffn_down_exps.weight") == 0) {
        component = moevm::ExpertComponent::Down;
        return true;
    }
    return false;
}

static bool same_shadow_ids_shape(
        const shadow_ids_slot & slot,
        shadow_ids_kind kind,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * ids_tensor) {
    if (slot.kind != kind || slot.backend != backend || slot.buft != buft || slot.type != ids_tensor->type || slot.n_dims != ggml_n_dims(ids_tensor) ||
            slot.bytes != ggml_nbytes(ids_tensor)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (slot.ne[i] != ids_tensor->ne[i]) {
            return false;
        }
    }
    return true;
}

static ggml_tensor * get_or_create_shadow_ids_tensor(
        moe_trace_state * state,
        shadow_ids_kind kind,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * ids_tensor) {
    for (shadow_ids_slot & slot : state->shadow_ids_slots) {
        if (same_shadow_ids_shape(slot, kind, backend, buft, ids_tensor)) {
            return slot.tensor;
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    const int n_dims = ggml_n_dims(ids_tensor);
    ggml_tensor * tensor = ggml_new_tensor(ctx, ids_tensor->type, n_dims, ids_tensor->ne);
    if (!tensor) {
        ggml_free(ctx);
        return nullptr;
    }
    const char * kind_name = kind == shadow_ids_kind::pool ? "pool" : kind == shadow_ids_kind::compact ? "compact" : "verbatim";
    ggml_format_name(tensor, "moevm_shadow_ids_%s_%zu", kind_name, state->shadow_ids_slots.size());

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_buffer_clear(buffer, 0);

    shadow_ids_slot slot;
    slot.kind = kind;
    slot.backend = backend;
    slot.buft = buft;
    slot.ctx = ctx;
    slot.buffer = buffer;
    slot.tensor = tensor;
    slot.type = ids_tensor->type;
    slot.n_dims = n_dims;
    slot.bytes = ggml_nbytes(ids_tensor);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        slot.ne[i] = ids_tensor->ne[i];
    }
    state->shadow_ids_slots.push_back(slot);
    return state->shadow_ids_slots.back().tensor;
}

static bool same_compact_weight_shape(
        const compact_weight_slot & slot,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * src_tensor,
        moevm::ExpertComponent component,
        int64_t capacity,
        size_t expert_size) {
    return slot.backend == backend &&
        slot.buft == buft &&
        slot.type == src_tensor->type &&
        slot.ne0 == src_tensor->ne[0] &&
        slot.ne1 == src_tensor->ne[1] &&
        slot.capacity == capacity &&
        slot.expert_size == expert_size &&
        slot.component == component;
}

static ggml_tensor * get_or_create_compact_weight_tensor(
        moe_trace_state * state,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * src_tensor,
        moevm::ExpertComponent component,
        int64_t capacity,
        size_t expert_size) {
    for (compact_weight_slot & slot : state->compact_weight_slots) {
        if (same_compact_weight_shape(slot, backend, buft, src_tensor, component, capacity, expert_size)) {
            return slot.tensor;
        }
    }

    if (ggml_n_dims(src_tensor) != 3 || capacity <= 0 || capacity == std::numeric_limits<int64_t>::max()) {
        return nullptr;
    }
    const int64_t alloc_capacity = capacity + 1;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, src_tensor->type, src_tensor->ne[0], src_tensor->ne[1], alloc_capacity);
    if (!tensor) {
        ggml_free(ctx);
        return nullptr;
    }
    if (tensor->nb[2] != expert_size) {
        ggml_free(ctx);
        return nullptr;
    }
    ggml_format_name(tensor, "moevm_compact_%s_%zu", src_tensor->name, state->compact_weight_slots.size());

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_buffer_clear(buffer, 0);

    compact_weight_slot slot;
    slot.backend = backend;
    slot.buft = buft;
    slot.ctx = ctx;
    slot.buffer = buffer;
    slot.tensor = tensor;
    slot.type = src_tensor->type;
    slot.ne0 = src_tensor->ne[0];
    slot.ne1 = src_tensor->ne[1];
    slot.capacity = capacity;
    slot.expert_size = expert_size;
    slot.component = component;
    state->compact_weight_slots.push_back(slot);
    return state->compact_weight_slots.back().tensor;
}

static bool same_pool_weight_shape(
        const pool_weight_slot & slot,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * src_tensor,
        uint32_t layer,
        moevm::ExpertComponent component,
        int64_t capacity,
        size_t expert_size) {
    return slot.backend == backend &&
        slot.buft == buft &&
        slot.type == src_tensor->type &&
        slot.layer == layer &&
        slot.ne0 == src_tensor->ne[0] &&
        slot.ne1 == src_tensor->ne[1] &&
        slot.capacity == capacity &&
        slot.expert_size == expert_size &&
        slot.component == component;
}

static pool_weight_slot * get_or_create_pool_weight_slot(
        moe_trace_state * state,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * src_tensor,
        uint32_t layer,
        moevm::ExpertComponent component,
        int64_t capacity,
        size_t expert_size) {
    for (pool_weight_slot & slot : state->pool_weight_slots) {
        if (same_pool_weight_shape(slot, backend, buft, src_tensor, layer, component, capacity, expert_size)) {
            return &slot;
        }
    }

    if (ggml_n_dims(src_tensor) != 3 || capacity <= 0 || capacity == std::numeric_limits<int64_t>::max()) {
        return nullptr;
    }
    const int64_t alloc_capacity = capacity + 1;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, src_tensor->type, src_tensor->ne[0], src_tensor->ne[1], alloc_capacity);
    if (!tensor) {
        ggml_free(ctx);
        return nullptr;
    }
    if (tensor->nb[2] != expert_size) {
        ggml_free(ctx);
        return nullptr;
    }
    ggml_format_name(tensor, "moevm_pool_l%u_%s_%zu", layer, src_tensor->name, state->pool_weight_slots.size());

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_buffer_clear(buffer, 0);

    pool_weight_slot slot;
    slot.backend = backend;
    slot.buft = buft;
    slot.ctx = ctx;
    slot.buffer = buffer;
    slot.tensor = tensor;
    slot.type = src_tensor->type;
    slot.layer = layer;
    slot.ne0 = src_tensor->ne[0];
    slot.ne1 = src_tensor->ne[1];
    slot.capacity = capacity;
    slot.expert_size = expert_size;
    slot.component = component;
    slot.loaded_expert_by_slot.assign((size_t) capacity, -1);
    state->pool_weight_slots.push_back(slot);
    return &state->pool_weight_slots.back();
}

static layer_pool_state * get_or_create_layer_pool(
        moe_trace_state * state,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        uint32_t layer,
        int64_t n_expert,
        int capacity) {
    if (n_expert <= 0 || capacity <= 0) {
        return nullptr;
    }

    for (layer_pool_state & pool : state->layer_pools) {
        if (pool.backend == backend && pool.buft == buft && pool.layer == layer && pool.n_expert == n_expert && pool.capacity == capacity) {
            return &pool;
        }
    }

    layer_pool_state pool;
    pool.backend = backend;
    pool.buft = buft;
    pool.layer = layer;
    pool.n_expert = n_expert;
    pool.capacity = capacity;
    pool.slot_by_expert.assign((size_t) n_expert, -1);
    pool.expert_by_slot.assign((size_t) capacity, -1);
    pool.last_used_by_slot.assign((size_t) capacity, 0);
    pool.last_epoch_by_slot.assign((size_t) capacity, 0);
    state->layer_pools.push_back(pool);
    return &state->layer_pools.back();
}

static bool plan_pool_slots(
        moe_trace_state * state,
        const layer_pool_state * pool,
        const int32_t * expert_ids,
        size_t expert_id_count,
        pool_slot_plan & plan,
        bool & blocked_by_pinned_slot) {
    if (!pool || !expert_ids || expert_id_count == 0 || expert_id_count > (size_t) pool->capacity) {
        return false;
    }

    plan = {};
    plan.slot_by_selected.assign(expert_id_count, -1);
    plan.slot_by_expert = pool->slot_by_expert;
    plan.expert_by_slot = pool->expert_by_slot;
    blocked_by_pinned_slot = false;
    std::vector<uint8_t> protected_slot((size_t) pool->capacity, 0);

    for (size_t i = 0; i < expert_id_count; ++i) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || expert >= pool->n_expert) {
            return false;
        }
        const int32_t slot = plan.slot_by_expert[(size_t) expert];
        if (slot >= 0) {
            plan.slot_by_selected[i] = slot;
            protected_slot[(size_t) slot] = 1;
        }
    }

    for (size_t i = 0; i < expert_id_count; ++i) {
        if (plan.slot_by_selected[i] >= 0) {
            continue;
        }

        int32_t chosen = -1;
        for (int32_t slot = 0; slot < pool->capacity; ++slot) {
            if (!protected_slot[(size_t) slot] && plan.expert_by_slot[(size_t) slot] < 0) {
                chosen = slot;
                break;
            }
        }

        if (chosen < 0) {
            uint64_t oldest_tick = std::numeric_limits<uint64_t>::max();
            for (int32_t slot = 0; slot < pool->capacity; ++slot) {
                if (protected_slot[(size_t) slot]) {
                    continue;
                }
                if (pool->last_epoch_by_slot[(size_t) slot] > state->moevm_pool_completed_epoch) {
                    blocked_by_pinned_slot = true;
                    ++state->moevm_pool_pinned_eviction_skips;
                    continue;
                }
                const uint64_t tick = pool->last_used_by_slot[(size_t) slot];
                if (tick < oldest_tick) {
                    oldest_tick = tick;
                    chosen = slot;
                }
            }
        }

        if (chosen < 0) {
            return false;
        }

        const int32_t old_expert = plan.expert_by_slot[(size_t) chosen];
        if (old_expert >= 0) {
            plan.slot_by_expert[(size_t) old_expert] = -1;
            ++plan.evictions;
        }

        const int32_t expert = expert_ids[i];
        plan.expert_by_slot[(size_t) chosen] = expert;
        plan.slot_by_expert[(size_t) expert] = chosen;
        plan.slot_by_selected[i] = chosen;
        protected_slot[(size_t) chosen] = 1;
    }

    for (int32_t slot : plan.slot_by_selected) {
        if (slot < 0 || slot >= pool->capacity) {
            return false;
        }
    }

    return true;
}

static void commit_pool_slot_plan(
        moe_trace_state * state,
        layer_pool_state * pool,
        const pool_slot_plan & plan,
        uint64_t patch_epoch) {
    pool->slot_by_expert = plan.slot_by_expert;
    pool->expert_by_slot = plan.expert_by_slot;
    state->moevm_pool_evictions += plan.evictions;
    for (int32_t slot : plan.slot_by_selected) {
        pool->last_used_by_slot[(size_t) slot] = ++pool->tick;
        pool->last_epoch_by_slot[(size_t) slot] = patch_epoch;
    }
}

static void moevm_inject_delay(moe_trace_state * state, uint64_t salt) {
    if (!state || state->moevm_inject_delay_max_us == 0) {
        return;
    }
    uint64_t x = state->moevm_delay_state ^ salt;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state->moevm_delay_state = x;
    const uint64_t us = (x * 2685821657736338717ull) % ((uint64_t) state->moevm_inject_delay_max_us + 1);
    if (us == 0) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(us));
    ++state->moevm_delay_injections;
    state->moevm_delay_us += us;
}

static void maybe_force_pinned_fence_once(
        moe_trace_state * state,
        layer_pool_state * pool,
        const int32_t * expert_ids,
        size_t expert_id_count) {
    if (!state || !pool || !expert_ids || !state->moevm_test_force_pinned_fence ||
            state->moevm_test_force_pinned_fence_done || pool->capacity <= 0) {
        return;
    }

    std::vector<uint8_t> protected_slot((size_t) pool->capacity, 0);
    bool needs_eviction = false;
    bool has_free_slot = false;
    for (size_t i = 0; i < expert_id_count; ++i) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || expert >= pool->n_expert) {
            return;
        }
        const int32_t slot = pool->slot_by_expert[(size_t) expert];
        if (slot >= 0) {
            protected_slot[(size_t) slot] = 1;
        } else {
            needs_eviction = true;
        }
    }
    if (!needs_eviction) {
        return;
    }
    for (int32_t slot = 0; slot < pool->capacity; ++slot) {
        if (!protected_slot[(size_t) slot] && pool->expert_by_slot[(size_t) slot] < 0) {
            has_free_slot = true;
            break;
        }
    }
    if (has_free_slot) {
        return;
    }

    const uint64_t synthetic_inflight_epoch = ++state->moevm_pool_submit_epoch;
    bool pinned_any = false;
    for (int32_t slot = 0; slot < pool->capacity; ++slot) {
        if (!protected_slot[(size_t) slot]) {
            pool->last_epoch_by_slot[(size_t) slot] = synthetic_inflight_epoch;
            pinned_any = true;
        }
    }
    if (pinned_any) {
        state->moevm_test_force_pinned_fence_done = true;
    }
}

static bool self_test_pool_pinning() {
    {
        moe_trace_state state;
        state.moevm_pool_completed_epoch = 4;

        layer_pool_state pool;
        pool.layer = 0;
        pool.n_expert = 8;
        pool.capacity = 3;
        pool.tick = 3;
        pool.slot_by_expert.assign(8, -1);
        pool.expert_by_slot = {0, 1, 2};
        pool.last_used_by_slot = {1, 2, 3};
        pool.last_epoch_by_slot = {5, 3, 2};
        pool.slot_by_expert[0] = 0;
        pool.slot_by_expert[1] = 1;
        pool.slot_by_expert[2] = 2;

        const int32_t ids[] = {3, 4};
        pool_slot_plan plan;
        bool blocked = false;
        if (!plan_pool_slots(&state, &pool, ids, 2, plan, blocked)) {
            LOG_ERR("pool pinning self-test failed: expected plan success with alternate evictable slots\n");
            return false;
        }
        if (plan.slot_by_selected.size() != 2 || plan.slot_by_selected[0] == 0 || plan.slot_by_selected[1] == 0) {
            LOG_ERR("pool pinning self-test failed: pinned slot was selected for eviction\n");
            return false;
        }
        if (!blocked || state.moevm_pool_pinned_eviction_skips == 0 || plan.evictions != 2) {
            LOG_ERR("pool pinning self-test failed: pinned skip counters/evictions did not match expectations\n");
            return false;
        }
        commit_pool_slot_plan(&state, &pool, plan, 6);
        if (pool.slot_by_expert[3] < 0 || pool.slot_by_expert[4] < 0 || pool.slot_by_expert[0] != 0) {
            LOG_ERR("pool pinning self-test failed: committed plan corrupted pool mapping\n");
            return false;
        }
    }

    {
        moe_trace_state state;
        state.moevm_defer_input_wait = true;
        state.moevm_pool_completed_epoch = 10;
        state.moevm_pool_submit_epoch = 12;

        layer_pool_state pool;
        pool.layer = 0;
        pool.n_expert = 4;
        pool.capacity = 2;
        pool.tick = 2;
        pool.slot_by_expert.assign(4, -1);
        pool.expert_by_slot = {0, 1};
        pool.last_used_by_slot = {1, 2};
        pool.last_epoch_by_slot = {12, 12};
        pool.slot_by_expert[0] = 0;
        pool.slot_by_expert[1] = 1;

        const int32_t ids[] = {2};
        pool_slot_plan plan;
        bool blocked = false;
        if (plan_pool_slots(&state, &pool, ids, 1, plan, blocked)) {
            LOG_ERR("pool pinning self-test failed: expected no evictable slot while both slots are pinned\n");
            return false;
        }
        if (!blocked || state.moevm_pool_pinned_eviction_skips != 2) {
            LOG_ERR("pool pinning self-test failed: no-evictable case did not report pinned slots\n");
            return false;
        }
        state.moevm_pool_completed_epoch = state.moevm_pool_submit_epoch;
        blocked = false;
        if (!plan_pool_slots(&state, &pool, ids, 1, plan, blocked)) {
            LOG_ERR("pool pinning self-test failed: expected plan success after completed epoch advanced\n");
            return false;
        }
        if (plan.slot_by_selected.size() != 1 || plan.slot_by_selected[0] < 0 || plan.slot_by_selected[0] >= pool.capacity) {
            LOG_ERR("pool pinning self-test failed: invalid retry slot after epoch advance\n");
            return false;
        }
    }

    {
        ggml_init_params params = {
            /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERR("pool pinning self-test failed: could not create ggml context\n");
            return false;
        }
        int64_t ne[2] = {3, 2};
        ggml_tensor * ids_tensor = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, ne);
        ggml_tensor * shadow = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, ne);
        if (!ids_tensor || !shadow) {
            ggml_free(ctx);
            LOG_ERR("pool pinning self-test failed: could not create remap tensors\n");
            return false;
        }
        ids_tensor->nb[0] = sizeof(int32_t);
        ids_tensor->nb[1] = 4 * sizeof(int32_t);
        shadow->nb[0] = sizeof(int32_t);
        shadow->nb[1] = 3 * sizeof(int32_t);

        int32_t ids_data[8] = {2, 0, 1, 777, 1, 2, 0, 888};
        int32_t shadow_data[6] = {};
        shadow->data = shadow_data;
        std::vector<int32_t> remap = {10, 11, 12};
        std::vector<int32_t> shadow_tmp;
        const bool remap_ok = build_shadow_ids_remap_i32(ids_tensor, ids_data, 8, shadow, remap, 3, shadow_tmp);
        ggml_free(ctx);
        if (!remap_ok || shadow_tmp.size() != 6 ||
                shadow_tmp[0] != 12 || shadow_tmp[1] != 10 || shadow_tmp[2] != 11 ||
                shadow_tmp[3] != 11 || shadow_tmp[4] != 12 || shadow_tmp[5] != 10) {
            LOG_ERR("pool pinning self-test failed: stride-aware shadow remap produced wrong output\n");
            return false;
        }
    }

    LOG_INF("MoEVM pool pinning self-test passed\n");
    return true;
}

static resident_weight_cache_slot * get_or_create_resident_weight_cache_slot(
        moe_trace_state * state,
        ggml_tensor * dst_tensor,
        const ggml_tensor * src_tensor,
        uint32_t layer,
        moevm::ExpertComponent component,
        int64_t n_expert,
        size_t expert_size) {
    if (!state || !dst_tensor || !src_tensor || n_expert <= 0) {
        return nullptr;
    }

    for (resident_weight_cache_slot & slot : state->resident_weight_cache_slots) {
        if (slot.dst_tensor != dst_tensor || slot.dst_buffer != dst_tensor->buffer || slot.dst_data != dst_tensor->data) {
            continue;
        }

        const bool same_source = slot.src_tensor == src_tensor &&
            slot.layer == layer &&
            slot.component == component &&
            slot.n_expert == n_expert &&
            slot.expert_size == expert_size;
        if (!same_source) {
            slot.src_tensor = src_tensor;
            slot.layer = layer;
            slot.component = component;
            slot.n_expert = n_expert;
            slot.expert_size = expert_size;
            slot.loaded_by_expert.assign((size_t) n_expert, 0);
            ++state->moevm_resident_cache_resets;
        }
        return &slot;
    }

    resident_weight_cache_slot slot;
    slot.dst_tensor = dst_tensor;
    slot.dst_buffer = dst_tensor->buffer;
    slot.dst_data = dst_tensor->data;
    slot.src_tensor = src_tensor;
    slot.layer = layer;
    slot.component = component;
    slot.n_expert = n_expert;
    slot.expert_size = expert_size;
    slot.loaded_by_expert.assign((size_t) n_expert, 0);
    state->resident_weight_cache_slots.push_back(slot);
    return &state->resident_weight_cache_slots.back();
}

static bool moevm_weight_copy_cb(
        ggml_backend_t             dst_backend,
        ggml_tensor *              dst_tensor,
        const ggml_tensor *        src_tensor,
        const int32_t *            expert_ids,
        size_t                     expert_id_count,
        size_t                     expert_size,
        ggml_tensor *              node,
        const ggml_tensor *        ids_tensor,
        const int32_t *            ids_data,
        size_t                     ids_data_count,
        bool                       ids_synchronized,
        void *                     user_data) {
    auto * state = (moe_trace_state *) user_data;
    if (!state || !state->runtime) {
        return false;
    }

    ++state->moevm_copy_hook_calls;
    if (state->moevm_defer_input_wait && ids_synchronized) {
        state->moevm_pool_completed_epoch = state->moevm_pool_submit_epoch;
        ++state->moevm_pool_epoch_drains;
    }

    uint32_t layer = 0;
    moevm::ExpertComponent component = moevm::ExpertComponent::Gate;
    if (!parse_routed_tensor_component(src_tensor->name, layer, component)) {
        return false;
    }

    try {
        std::vector<uint32_t> selected;
        selected.reserve(expert_id_count);
        for (size_t i = 0; i < expert_id_count; ++i) {
            selected.push_back((uint32_t) expert_ids[i]);
        }
        if (!selected.empty()) {
            if (state->copy_prefetch_cache.size() <= layer) {
                state->copy_prefetch_cache.resize((size_t) layer + 1);
            }
            copy_prefetch_cache_entry & cached = state->copy_prefetch_cache[layer];
            if (cached.decode_base == state->decode_base && cached.ids == selected) {
                ++state->moevm_copy_prefetch_skips;
            } else {
                state->runtime->prefetch_candidates(layer, selected.data(), selected.size());
                cached.decode_base = state->decode_base;
                cached.ids = selected;
                ++state->moevm_copy_prefetches;
            }
        }
        bool pool_done = false;
        const bool wants_pool = state->moevm_pool_layer == -2 || (state->moevm_pool_layer >= 0 && (int) layer == state->moevm_pool_layer);
        if (wants_pool) {
            ++state->moevm_pool_attempts;
            bool pool_ok = false;
            if (expert_id_count > (size_t) state->moevm_pool_capacity) {
                ++state->moevm_fallback_k_overflow;
            } else if (node != nullptr && ids_tensor != nullptr && ids_data != nullptr && ids_tensor->type == GGML_TYPE_I32 &&
                    ggml_n_dims(src_tensor) == 3 && expert_id_count > 0) {
                const size_t ids_bytes = ggml_nbytes(ids_tensor);
                if (ids_data_count * sizeof(int32_t) >= ids_bytes) {
                    ggml_backend_buffer_type_t buft = dst_tensor->buffer ? ggml_backend_buffer_get_type(dst_tensor->buffer) : ggml_backend_get_default_buffer_type(dst_backend);
                    layer_pool_state * layer_pool = get_or_create_layer_pool(
                            state, dst_backend, buft, layer, src_tensor->ne[2], state->moevm_pool_capacity);
                    pool_weight_slot * pool_slot = get_or_create_pool_weight_slot(
                            state, dst_backend, buft, src_tensor, layer, component, state->moevm_pool_capacity, expert_size);
                    ggml_tensor * shadow = get_or_create_shadow_ids_tensor(state, shadow_ids_kind::pool, dst_backend, buft, ids_tensor);
                    const bool alloc_ok = layer_pool != nullptr && pool_slot != nullptr && pool_slot->tensor != nullptr && pool_slot->tensor->data != nullptr &&
                        shadow != nullptr && shadow->data != nullptr;
                    if (alloc_ok) {
                        pool_slot_plan plan;
                        bool blocked_by_pinned_slot = false;
                        maybe_force_pinned_fence_once(state, layer_pool, expert_ids, expert_id_count);
                        pool_ok = plan_pool_slots(state, layer_pool, expert_ids, expert_id_count, plan, blocked_by_pinned_slot);
                        if (!pool_ok && blocked_by_pinned_slot && state->moevm_defer_input_wait) {
                            ggml_backend_synchronize(dst_backend);
                            state->moevm_pool_completed_epoch = state->moevm_pool_submit_epoch;
                            ++state->moevm_pool_forced_fences;
                            blocked_by_pinned_slot = false;
                            pool_ok = plan_pool_slots(state, layer_pool, expert_ids, expert_id_count, plan, blocked_by_pinned_slot);
                        }
                        if (!pool_ok) {
                            ++state->moevm_pool_fallback_plan;
                        }

                        std::vector<int32_t> shadow_ids_tmp;
                        if (pool_ok) {
                            std::vector<int32_t> remap((size_t) src_tensor->ne[2], -1);
                            for (size_t j = 0; j < expert_id_count; ++j) {
                                const int32_t expert = expert_ids[j];
                                if (expert >= 0 && expert < src_tensor->ne[2]) {
                                    remap[(size_t) expert] = plan.slot_by_selected[j];
                                }
                            }

                            const bool remap_ok = build_shadow_ids_remap_i32(
                                    ids_tensor,
                                    ids_data,
                                    ids_data_count,
                                    shadow,
                                    remap,
                                    src_tensor->ne[2],
                                    shadow_ids_tmp);
                            if (!remap_ok) {
                                ++state->moevm_pool_fallback_remap;
                                pool_ok = false;
                            }
                        }

                        std::vector<int32_t> touched_slots;
                        if (pool_ok) {
                            pool_slot->tensor->ne[2] = state->moevm_pool_capacity;
                            for (size_t j = 0; j < expert_id_count; ++j) {
                                const uint32_t expert = (uint32_t) expert_ids[j];
                                const int32_t slot = plan.slot_by_selected[j];
                                if (slot < 0 || slot >= state->moevm_pool_capacity) {
                                    ++state->moevm_pool_fallback_fill;
                                    pool_ok = false;
                                    break;
                                }
                                if (pool_slot->loaded_expert_by_slot[(size_t) slot] == (int32_t) expert) {
                                    ++state->moevm_pool_component_hits;
                                    continue;
                                }
                                try {
                                    moevm_inject_delay(state, ((uint64_t) layer << 32) ^ ((uint64_t) expert << 2) ^ (uint64_t) component);
                                    moevm::RecordView record = state->runtime->wait_selected(layer, expert);
                                    moevm_record_guard record_guard(state->runtime.get(), record);
                                    moevm::ComponentView view = state->runtime->component_view(record, component);
                                    if (view.bytes != expert_size) {
                                        moevm_set_runtime_error(state, std::string("MoEVM component size mismatch for pool tensor ") + src_tensor->name);
                                        ++state->moevm_pool_fallback_fill;
                                        pool_ok = false;
                                        break;
                                    }
                                    std::memcpy((uint8_t *) pool_slot->tensor->data + (size_t) slot * expert_size, view.data, view.bytes);
                                    pool_slot->loaded_expert_by_slot[(size_t) slot] = (int32_t) expert;
                                    touched_slots.push_back(slot);
                                    ++state->moevm_pool_component_misses;
                                    ++state->moevm_copy_memcpy_ops;
                                    ++state->moevm_copy_experts;
                                    state->moevm_copy_bytes += view.bytes;
                                } catch (const std::exception & e) {
                                    moevm_set_runtime_error(state, e.what());
                                    ++state->moevm_pool_fallback_fill;
                                    pool_ok = false;
                                    break;
                                }
                            }
                        }

                        if (pool_ok) {
                            std::memcpy(shadow->data, shadow_ids_tmp.data(), ggml_nbytes(shadow));
                            const uint64_t patch_epoch = state->moevm_defer_input_wait
                                ? ++state->moevm_pool_submit_epoch
                                : state->moevm_pool_completed_epoch;
                            commit_pool_slot_plan(state, layer_pool, plan, patch_epoch);
                            node->src[0] = pool_slot->tensor;
                            node->src[2] = shadow;
                            pool_done = true;
                        } else {
                            for (int32_t slot : touched_slots) {
                                if (slot >= 0 && slot < (int32_t) pool_slot->loaded_expert_by_slot.size()) {
                                    pool_slot->loaded_expert_by_slot[(size_t) slot] = -1;
                                }
                            }
                        }
                    } else {
                        ++state->moevm_pool_fallback_alloc;
                        if (layer_pool == nullptr) {
                            ++state->moevm_pool_fallback_alloc_layer;
                        }
                        if (pool_slot == nullptr || pool_slot->tensor == nullptr) {
                            ++state->moevm_pool_fallback_alloc_pool;
                        } else if (pool_slot->tensor->data == nullptr) {
                            ++state->moevm_pool_fallback_alloc_pool_data;
                        }
                        if (shadow == nullptr) {
                            ++state->moevm_pool_fallback_alloc_shadow;
                        } else {
                            if (shadow->data == nullptr) {
                                ++state->moevm_pool_fallback_alloc_shadow_data;
                            }
                        }
                    }
                } else {
                    ++state->moevm_pool_fallback_ids_read;
                }
            } else {
                ++state->moevm_pool_fallback_shape;
            }
            if (pool_done) {
                ++state->moevm_pool_applied;
            } else {
                ++state->moevm_pool_fallback;
            }
        }

        bool compact_done = false;
        const bool wants_compact = state->moevm_compact_layer == -2 || (state->moevm_compact_layer >= 0 && (int) layer == state->moevm_compact_layer);
        if (!pool_done && wants_compact) {
            ++state->moevm_compact_attempts;
            bool compact_ok = false;
            if (expert_id_count > (size_t) state->moevm_compact_k_max) {
                ++state->moevm_fallback_k_overflow;
            } else if (node != nullptr && ids_tensor != nullptr && ids_data != nullptr && ids_tensor->type == GGML_TYPE_I32 &&
                    ggml_n_dims(src_tensor) == 3 && expert_id_count > 0) {
                const size_t ids_bytes = ggml_nbytes(ids_tensor);
                if (ids_data_count * sizeof(int32_t) >= ids_bytes) {
                    ggml_backend_buffer_type_t buft = dst_tensor->buffer ? ggml_backend_buffer_get_type(dst_tensor->buffer) : ggml_backend_get_default_buffer_type(dst_backend);
                    ggml_tensor * compact = get_or_create_compact_weight_tensor(
                            state, dst_backend, buft, src_tensor, component, state->moevm_compact_k_max, expert_size);
                    ggml_tensor * shadow = get_or_create_shadow_ids_tensor(state, shadow_ids_kind::compact, dst_backend, buft, ids_tensor);
                    if (compact != nullptr && compact->data != nullptr && shadow != nullptr && shadow->data != nullptr) {
                        std::vector<int32_t> remap((size_t) src_tensor->ne[2], -1);
                        for (size_t j = 0; j < expert_id_count; ++j) {
                            const int32_t expert = expert_ids[j];
                            if (expert >= 0 && expert < src_tensor->ne[2]) {
                                remap[(size_t) expert] = (int32_t) j;
                            }
                        }

                        compact_ok = true;
                        std::vector<int32_t> shadow_ids_tmp;
                        compact_ok = build_shadow_ids_remap_i32(
                                ids_tensor,
                                ids_data,
                                ids_data_count,
                                shadow,
                                remap,
                                src_tensor->ne[2],
                                shadow_ids_tmp);

                        if (compact_ok) {
                            compact->ne[2] = (int64_t) expert_id_count;
                            for (size_t j = 0; j < expert_id_count; ++j) {
                                const uint32_t expert = (uint32_t) expert_ids[j];
                                moevm_inject_delay(state, ((uint64_t) layer << 32) ^ ((uint64_t) expert << 2) ^ (uint64_t) component ^ 0xc0ffac7ull);
                                moevm::RecordView record = state->runtime->wait_selected(layer, expert);
                                moevm_record_guard record_guard(state->runtime.get(), record);
                                moevm::ComponentView view = state->runtime->component_view(record, component);
                                if (view.bytes != expert_size) {
                                    moevm_set_runtime_error(state, std::string("MoEVM component size mismatch for compact tensor ") + src_tensor->name);
                                    return false;
                                }
                                std::memcpy((uint8_t *) compact->data + j * expert_size, view.data, view.bytes);
                                ++state->moevm_copy_memcpy_ops;
                                ++state->moevm_copy_experts;
                                state->moevm_copy_bytes += view.bytes;
                            }
                            std::memcpy(shadow->data, shadow_ids_tmp.data(), ggml_nbytes(shadow));
                            node->src[0] = compact;
                            node->src[2] = shadow;
                            compact_done = true;
                        }
                    }
                }
            }
            if (compact_done) {
                ++state->moevm_compact_applied;
            } else {
                ++state->moevm_compact_fallback;
            }
        }

        if (pool_done || compact_done) {
            ++state->moevm_copy_hook_handled;
            return true;
        }
        if (state->moevm_defer_input_wait) {
            return false;
        }
        if (state->moevm_shadow_ids) {
            ++state->moevm_shadow_ids_attempts;
            bool shadow_ok = false;
            if (node != nullptr && ids_tensor != nullptr && ids_data != nullptr && ids_tensor->type == GGML_TYPE_I32 && ggml_is_contiguous(ids_tensor)) {
                const size_t ids_bytes = ggml_nbytes(ids_tensor);
                if (ids_data_count * sizeof(int32_t) >= ids_bytes) {
                    ggml_backend_buffer_type_t buft = dst_tensor->buffer ? ggml_backend_buffer_get_type(dst_tensor->buffer) : ggml_backend_get_default_buffer_type(dst_backend);
                    ggml_tensor * shadow = get_or_create_shadow_ids_tensor(state, shadow_ids_kind::verbatim, dst_backend, buft, ids_tensor);
                    if (shadow != nullptr && shadow->data != nullptr && ggml_nbytes(shadow) == ids_bytes) {
                        std::memcpy(shadow->data, ids_data, ids_bytes);
                        node->src[2] = shadow;
                        shadow_ok = true;
                    }
                }
            }
            if (shadow_ok) {
                ++state->moevm_shadow_ids_applied;
            } else {
                ++state->moevm_shadow_ids_fallback;
            }
        }
        if (state->moevm_copy_mode_value == moevm_copy_mode::memcpy && dst_tensor->data == nullptr) {
            moevm_set_runtime_error(state, "MoEVM memcpy copy requested but destination tensor has no host-visible data");
            return false;
        }
        const size_t dst_bytes = ggml_nbytes(dst_tensor);
        resident_weight_cache_slot * resident_cache = state->moevm_resident_cache
            ? get_or_create_resident_weight_cache_slot(state, dst_tensor, src_tensor, layer, component, src_tensor->ne[2], expert_size)
            : nullptr;
        for (size_t i = 0; i < expert_id_count; ++i) {
            const uint32_t expert = (uint32_t) expert_ids[i];
            if (resident_cache != nullptr && expert < resident_cache->loaded_by_expert.size() && resident_cache->loaded_by_expert[(size_t) expert]) {
                ++state->moevm_resident_cache_hits;
                continue;
            }
            moevm_inject_delay(state, ((uint64_t) layer << 32) ^ ((uint64_t) expert << 2) ^ (uint64_t) component ^ 0x1e9ac7ull);
            moevm::RecordView record = state->runtime->wait_selected(layer, expert);
            moevm_record_guard record_guard(state->runtime.get(), record);
            moevm::ComponentView view = state->runtime->component_view(record, component);
            if (view.bytes != expert_size) {
                moevm_set_runtime_error(state, std::string("MoEVM component size mismatch for tensor ") + src_tensor->name);
                return false;
            }
            const size_t expert_offset = expert * expert_size;
            if (expert_offset > dst_bytes || view.bytes > dst_bytes - expert_offset) {
                moevm_set_runtime_error(state, std::string("MoEVM destination tensor bounds check failed for tensor ") + src_tensor->name);
                return false;
            }
            if (state->moevm_copy_mode_value == moevm_copy_mode::memcpy) {
                std::memcpy((uint8_t *) dst_tensor->data + expert_offset, view.data, view.bytes);
                ++state->moevm_copy_memcpy_ops;
            } else {
                ggml_backend_tensor_set_async(dst_backend, dst_tensor, view.data, expert_offset, view.bytes);
                ++state->moevm_copy_async_ops;
            }
            ++state->moevm_copy_experts;
            state->moevm_copy_bytes += view.bytes;
            if (resident_cache != nullptr && expert < resident_cache->loaded_by_expert.size()) {
                resident_cache->loaded_by_expert[(size_t) expert] = 1;
                ++state->moevm_resident_cache_misses;
            }
        }
    } catch (const std::exception & e) {
        moevm_set_runtime_error(state, e.what());
        return false;
    }

    ++state->moevm_copy_hook_handled;
    return true;
}

static bool moe_trace_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    int layer = -1;
    const trace_tensor_kind kind = parse_layer(t->name, layer);
    auto * state = (moe_trace_state *) user_data;
    const bool match = kind == trace_tensor_kind::topk ||
        (kind == trace_tensor_kind::candidates && state && state->candidate_limit > 0);
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }

    if (!state || !state->out) {
        return true;
    }

    const int64_t n_tokens = t->ne[1];
    if (state->last_layer >= 0 && layer < state->last_layer) {
        state->chunk_base += state->last_chunk_tokens;
    }
    state->last_layer = layer;
    state->last_chunk_tokens = n_tokens;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data = nullptr;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        const size_t n_bytes = ggml_nbytes(t);
        state->scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, state->scratch.data(), 0, n_bytes);
        data = state->scratch.data();
    }

    if (!data) {
        return true;
    }

    const int64_t n_expert_used = kind == trace_tensor_kind::candidates && state->candidate_limit > 0
        ? std::min<int64_t>(t->ne[0], state->candidate_limit)
        : t->ne[0];
    std::vector<uint32_t> prefetch_ids;
    if (state->runtime && (kind == trace_tensor_kind::topk || kind == trace_tensor_kind::candidates)) {
        const int64_t prefetch_count = std::min<int64_t>(n_expert_used, state->moevm_prefetch_topk);
        prefetch_ids.reserve((size_t) prefetch_count);
        for (int64_t k = 0; k < prefetch_count; ++k) {
            const int64_t id = get_i64(data, t, k, 0);
            if (id >= 0) {
                prefetch_ids.push_back((uint32_t) id);
            }
        }
        if (!prefetch_ids.empty()) {
            try {
                state->runtime->prefetch_candidates((uint32_t) layer, prefetch_ids.data(), prefetch_ids.size());
                ++state->moevm_prefetch_records;
            } catch (const std::exception & e) {
                moevm_set_runtime_error(state, e.what());
            }
        }
    }
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        if (state->runtime && tok > 0 && (kind == trace_tensor_kind::topk || kind == trace_tensor_kind::candidates)) {
            const int64_t prefetch_count = std::min<int64_t>(n_expert_used, state->moevm_prefetch_topk);
            prefetch_ids.clear();
            prefetch_ids.reserve((size_t) prefetch_count);
            for (int64_t k = 0; k < prefetch_count; ++k) {
                const int64_t id = get_i64(data, t, k, tok);
                if (id >= 0) {
                    prefetch_ids.push_back((uint32_t) id);
                }
            }
            if (!prefetch_ids.empty()) {
                try {
                    state->runtime->prefetch_candidates((uint32_t) layer, prefetch_ids.data(), prefetch_ids.size());
                    ++state->moevm_prefetch_records;
                } catch (const std::exception & e) {
                    moevm_set_runtime_error(state, e.what());
                }
            }
        }
        if (kind == trace_tensor_kind::candidates) {
            std::fprintf(state->out, "{\"kind\":\"candidates\",\"token\":%lld,\"layer\":%d,\"candidates\":[",
                    (long long) (state->chunk_base + tok), layer);
        } else {
            std::fprintf(state->out, "{\"token\":%lld,\"layer\":%d,\"experts\":[",
                    (long long) (state->chunk_base + tok), layer);
        }
        for (int64_t k = 0; k < n_expert_used; ++k) {
            if (k) {
                std::fputc(',', state->out);
            }
            std::fprintf(state->out, "%lld", (long long) get_i64(data, t, k, tok));
        }
        std::fprintf(state->out, "]}\n");
        state->records++;
    }
    std::fflush(state->out);

    return true;
}

static void moevm_prepare_decode_state(moe_trace_state & state, int64_t decode_base, int64_t chunk_tokens) {
    state.decode_base = decode_base;
    state.chunk_base = decode_base;
    state.last_chunk_tokens = chunk_tokens;
    state.last_layer = -1;
}

static bool run_trace(llama_context * ctx, const common_params & params, moe_trace_state & state) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt; provide -p or -f\n", __func__);
        return false;
    }
    const int32_t n_predict = params.n_predict > 0 ? params.n_predict : 0;

    const int n_batch = llama_n_batch(ctx);
    if (n_batch <= 0) {
        LOG_ERR("%s: invalid n_batch = %d\n", __func__, n_batch);
        return false;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int64_t n_past = 0;
    const std::vector<llama_seq_id> seq_ids{0};

    while (n_past < (int64_t) tokens.size()) {
        common_batch_clear(batch);
        const int64_t n_eval = std::min<int64_t>(n_batch, (int64_t) tokens.size() - n_past);
        for (int64_t i = 0; i < n_eval; ++i) {
            const bool needs_logits = n_predict > 0 && n_past + i == (int64_t) tokens.size() - 1;
            common_batch_add(batch, tokens[n_past + i], n_past + i, seq_ids, needs_logits);
        }

        moevm_prepare_decode_state(state, n_past, n_eval);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed at token offset %lld\n", __func__, (long long) n_past);
            llama_batch_free(batch);
            return false;
        }
        n_past += n_eval;
        LOG_INF("traced %lld / %zu prompt tokens, records=%lld\n",
                (long long) n_past, tokens.size(), (long long) state.records);
    }

    if (n_predict > 0) {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;
        llama_sampler * sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        const int64_t t_decode_start = ggml_time_us();
        int32_t n_decoded = 0;
        for (; n_decoded < n_predict; ++n_decoded) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            if (!state.write_logits_row((uint32_t) n_decoded, token, logits)) {
                LOG_ERR("%s: failed to write logits row %d\n", __func__, n_decoded);
                llama_sampler_free(sampler);
                llama_batch_free(batch);
                return false;
            }
            llama_sampler_accept(sampler, token);
            state.generated_tokens.push_back(token);
            if (!params.sampling.ignore_eos && llama_vocab_is_eog(vocab, token)) {
                LOG_INF("decode stopped at EOG after %d / %d tokens\n", n_decoded, n_predict);
                break;
            }

            common_batch_clear(batch);
            common_batch_add(batch, token, n_past, seq_ids, true);

            moevm_prepare_decode_state(state, n_past, 1);
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed during generation at token offset %lld\n", __func__, (long long) n_past);
                llama_sampler_free(sampler);
                llama_batch_free(batch);
                return false;
            }
            ++n_past;
        }
        const int64_t t_decode_end = ggml_time_us();
        const double decode_s = (double) (t_decode_end - t_decode_start) / 1000000.0;
        LOG_INF("decoded %d / %d tokens in %.3fs, %.2f tok/s, records=%lld\n",
                n_decoded, n_predict, decode_s, decode_s > 0.0 ? (double) n_decoded / decode_s : 0.0, (long long) state.records);
        std::string token_ids = "generated_token_ids=[";
        for (size_t i = 0; i < state.generated_tokens.size(); ++i) {
            if (i) {
                token_ids += ",";
            }
            token_ids += std::to_string(state.generated_tokens[i]);
        }
        token_ids += "]";
        LOG_INF("%s\n", token_ids.c_str());
        llama_sampler_free(sampler);
    }

    llama_batch_free(batch);
    return true;
}

static bool run_speculative_trace(llama_context * ctx_tgt, common_params params, moe_trace_state & state) {
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    if (!params.speculative.has_dft()) {
        LOG_ERR("%s: speculative mode requires --spec-draft-model or --spec-draft-hf\n", __func__);
        return false;
    }

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    {
        const auto & params_spec = params.speculative.draft;
        auto params_dft = params;

        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return false;
        }

        auto cparams = common_context_params_to_llama(params_dft);
        cparams.ctx_other = ctx_tgt;
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
        if (ctx_dft == nullptr) {
            LOG_ERR("failed to init draft context, '%s'\n", params_dft.model.path.c_str());
            return false;
        }

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    }

    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt)       == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = (common_context_can_seq_rm(ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    if (inp.empty()) {
        LOG_ERR("%s: empty prompt; provide -p or -f\n", __func__);
        return false;
    }
    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));
        return false;
    }
    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));
        return false;
    }

    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    bool has_eos = false;
    llama_seq_id seq_id = 0;

    const int64_t t_enc_start = ggml_time_us();

    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    const auto & params_spec = params.speculative;
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const int64_t n_prompt_eval = (int64_t) inp.size() - 1;
    if (n_prompt_eval > 0) {
        common_batch_clear(batch_tgt);
        for (int64_t i = 0; i < n_prompt_eval; ++i) {
            common_batch_add(batch_tgt, inp[(size_t) i], (llama_pos) i, { seq_id }, false);
        }
        moevm_prepare_decode_state(state, 0, n_prompt_eval);
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: target prompt decode failed\n", __func__);
            common_speculative_free(spec);
            llama_batch_free(batch_tgt);
            return false;
        }
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s: speculative prompt process failed\n", __func__);
            common_speculative_free(spec);
            llama_batch_free(batch_tgt);
            return false;
        }
    }

    llama_token id_last = inp.back();

    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = (int) inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    size_t n_draft = 0;
    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    const int64_t t_enc_end = ggml_time_us();
    const int64_t t_dec_start = ggml_time_us();

    while (true) {
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ -1,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft,
            };
            common_speculative_draft(spec);
            n_draft = draft.size();

            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
        } else if (use_ckpt_tgt) {
            GGML_ASSERT(!ckpt.empty());
        }

        common_batch_clear(batch_tgt);
        const int target_decode_base = n_past;
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
        }

        moevm_prepare_decode_state(state, target_decode_base, batch_tgt.n_tokens);
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: target speculative decode failed at token offset %d\n", __func__, target_decode_base);
            common_speculative_free(spec);
            llama_batch_free(batch_tgt);
            return false;
        }

        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s: speculative process failed at token offset %d\n", __func__, target_decode_base);
            common_speculative_free(spec);
            llama_batch_free(batch_tgt);
            return false;
        }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        GGML_ASSERT(ids.size() > 0);

        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            draft = std::move(ids);

            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);

            ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        if (n_draft > 0) {
            common_speculative_accept(spec, seq_id, ids.size() - 1);
        }

        n_past    += ids.size() - 1;
        n_drafted += (int) n_draft;
        n_accept  += (int) ids.size() - 1;
        n_predict += (int) ids.size();

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            state.generated_tokens.push_back(id_last);

            if (!params.sampling.ignore_eos && llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
            LOG("%s", token_str.c_str());
        }

        draft.clear();

        llama_memory_seq_rm(llama_get_memory(ctx_tgt),       seq_id, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1);

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    const int64_t t_dec_end = ggml_time_us();
    const int n_input = (int) inp.size();
    const double enc_s = (double) (t_enc_end - t_enc_start) / 1000000.0;
    const double dec_s = (double) (t_dec_end - t_dec_start) / 1000000.0;

    LOG("\n\n");
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n",
            n_input, enc_s, enc_s > 0.0 ? n_input / enc_s : 0.0);
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n",
            n_predict, dec_s, dec_s > 0.0 ? n_predict / dec_s : 0.0);
    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);

    std::string token_ids = "generated_token_ids=[";
    for (size_t i = 0; i < state.generated_tokens.size(); ++i) {
        if (i) {
            token_ids += ",";
        }
        token_ids += std::to_string(state.generated_tokens[i]);
    }
    token_ids += "]";
    LOG_INF("%s\n", token_ids.c_str());

    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::vector<char *> filtered;
    std::string trace_out;
    std::string logits_out;
    std::string generated_out;
    int candidate_limit = 0;
    moevm_local_params moevm_params;
    if (!parse_local_arg(argc, argv, filtered, trace_out, logits_out, generated_out, candidate_limit, moevm_params)) {
        return 1;
    }
    if (moevm_params.self_test_pool_pinning) {
        return self_test_pool_pinning() ? 0 : 1;
    }

    common_params params;
    common_init();

    if (!common_params_parse((int) filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    const bool moevm_enabled = !moevm_params.index_path.empty() ||
        !moevm_params.component_index_path.empty() ||
        !moevm_params.sidecar_path.empty();
    const bool trace_enabled = !trace_out.empty();

    moe_trace_state state;
    state.candidate_limit = candidate_limit;
    state.moevm_prefetch_topk = moevm_params.prefetch_topk;
    state.moevm_copy_mode_value = moevm_params.copy_mode;
    state.moevm_shadow_ids = moevm_params.shadow_ids;
    state.moevm_compact_layer = moevm_params.compact_layer;
    state.moevm_compact_k_max = std::max(1, moevm_params.compact_k_max);
    state.moevm_pool_layer = moevm_params.pool_layer;
    state.moevm_pool_capacity = std::max(1, moevm_params.pool_capacity);
    state.moevm_resident_cache = moevm_params.resident_cache;
    state.moevm_defer_input_wait = moevm_params.defer_input_wait &&
        (moevm_params.pool_layer != -1 || moevm_params.compact_layer != -1);
    state.moevm_test_force_pinned_fence = moevm_params.test_force_pinned_fence;
    state.moevm_inject_delay_max_us = moevm_params.inject_delay_max_us;
    state.moevm_delay_state = moevm_params.inject_delay_seed;
    if (trace_enabled) {
        state.out = std::fopen(trace_out.c_str(), "w");
        if (!state.out) {
            LOG_ERR("failed to open trace output '%s': %s\n", trace_out.c_str(), std::strerror(errno));
            return 1;
        }
        params.cb_eval = moe_trace_cb_eval;
        params.cb_eval_user_data = &state;
    } else {
        LOG_INF("MoE trace disabled: running without cb_eval/readback tracing\n");
    }
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        if (state.out) {
            std::fclose(state.out);
        }
        return 1;
    }
    if (!logits_out.empty()) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (!state.open_logits_out(logits_out, (uint32_t) n_vocab)) {
            LOG_ERR("failed to open logits output '%s': %s\n", logits_out.c_str(), std::strerror(errno));
            if (state.out) {
                std::fclose(state.out);
            }
            llama_backend_free();
            return 1;
        }
        LOG_INF("logits dump enabled: path=%s n_vocab=%d\n", logits_out.c_str(), n_vocab);
    }

    if (moevm_enabled) {
        if (moevm_params.index_path.empty() || moevm_params.component_index_path.empty() || moevm_params.sidecar_path.empty()) {
            LOG_ERR("MoEVM requires --moevm-index, --moevm-component-index, and --moevm-sidecar\n");
            if (state.out) {
                std::fclose(state.out);
            }
            llama_backend_free();
            return 1;
        }
        try {
            moevm::RuntimeOptions runtime_options;
            runtime_options.index_path = moevm_params.index_path;
            runtime_options.component_index_path = moevm_params.component_index_path;
            runtime_options.sidecar_path = moevm_params.sidecar_path;
            runtime_options.arena_bytes = (uint64_t) (moevm_params.arena_gib * 1024.0 * 1024.0 * 1024.0);
            runtime_options.max_inflight_bytes = (uint64_t) (moevm_params.inflight_mib * 1024.0 * 1024.0);
            runtime_options.prefetch_threads = (uint32_t) moevm_params.threads;
            runtime_options.nocache = moevm_params.nocache;
            runtime_options.direct_mmap = moevm_params.direct_mmap;
            runtime_options.inject_delay_max_us = moevm_params.inject_delay_max_us;
            runtime_options.inject_delay_seed = moevm_params.inject_delay_seed;
            state.runtime.reset(new moevm::Runtime(runtime_options));
            state.runtime->start();
            ggml_backend_reset_moe_weight_copy_stats();
            ggml_backend_set_moe_weight_copy_debug(moevm_params.copy_debug);
            ggml_backend_set_moe_weight_copy_defer_input_wait(state.moevm_defer_input_wait);
            ggml_backend_set_moe_weight_copy_callback(moevm_weight_copy_cb, &state);
            LOG_INF("MoEVM enabled: arena=%.2fGiB inflight=%.2fMiB threads=%d prefetch_topk=%d copy_debug=%d direct_mmap=%d nocache=%d copy_mode=%s shadow_ids=%d compact_layer=%d compact_k_max=%d pool_layer=%d pool_capacity=%d resident_cache=%d defer_input_wait=%d inject_delay_us=%u test_force_pinned_fence=%d\n",
                    moevm_params.arena_gib, moevm_params.inflight_mib, moevm_params.threads, moevm_params.prefetch_topk, moevm_params.copy_debug, moevm_params.direct_mmap ? 1 : 0,
                    moevm_params.nocache ? 1 : 0,
                    moevm_copy_mode_name(moevm_params.copy_mode),
                    moevm_params.shadow_ids ? 1 : 0,
                    moevm_params.compact_layer,
                    std::max(1, moevm_params.compact_k_max),
                    moevm_params.pool_layer,
                    std::max(1, moevm_params.pool_capacity),
                    moevm_params.resident_cache ? 1 : 0,
                    state.moevm_defer_input_wait ? 1 : 0,
                    moevm_params.inject_delay_max_us,
                    state.moevm_test_force_pinned_fence ? 1 : 0);
        } catch (const std::exception & e) {
            LOG_ERR("failed to initialize MoEVM runtime: %s\n", e.what());
            if (state.out) {
                std::fclose(state.out);
            }
            llama_backend_free();
            return 1;
        }
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    const bool run_speculative = moevm_params.speculative || params.speculative.has_dft();
    const bool ok = run_speculative ? run_speculative_trace(ctx, params, state) : run_trace(ctx, params, state);
    if (ok && !generated_out.empty()) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string text = common_detokenize(vocab, state.generated_tokens, false);
        FILE * generated_file = std::fopen(generated_out.c_str(), "wb");
        if (!generated_file) {
            LOG_ERR("failed to open generated output '%s': %s\n", generated_out.c_str(), std::strerror(errno));
        } else {
            std::fwrite(text.data(), 1, text.size(), generated_file);
            std::fclose(generated_file);
            LOG_INF("generated text written: path=%s bytes=%zu\n", generated_out.c_str(), text.size());
        }
    }
    ggml_backend_set_moe_weight_copy_callback(nullptr, nullptr);
    ggml_backend_set_moe_weight_copy_debug(0);
    ggml_backend_set_moe_weight_copy_defer_input_wait(false);
    if (state.runtime) {
        ggml_backend_moe_weight_copy_stats copy_stats = {};
        ggml_backend_get_moe_weight_copy_stats(&copy_stats);
        state.runtime->stop();
        const moevm::RuntimeMetrics m = state.runtime->metrics();
        uint64_t moevm_runtime_errors = 0;
        std::string runtime_error_first;
        std::string runtime_error_last;
        {
            std::lock_guard<std::mutex> lock(state.runtime_error_mu);
            moevm_runtime_errors = state.moevm_runtime_errors;
            runtime_error_first = state.runtime_error;
            runtime_error_last = state.runtime_error_last;
        }
        LOG_INF("MoEVM prefetch_records=%llu copy_prefetches=%llu copy_prefetch_skips=%llu copy_hook_calls=%llu handled=%llu copied_experts=%llu copied_bytes=%llu copy_mode=%s async_ops=%llu memcpy_ops=%llu shadow_ids_attempts=%llu shadow_ids_applied=%llu shadow_ids_fallback=%llu compact_attempts=%llu compact_applied=%llu compact_fallback=%llu fallback_k_overflow=%llu pool_attempts=%llu pool_applied=%llu pool_fallback=%llu pool_component_hits=%llu pool_component_misses=%llu pool_evictions=%llu pool_fallback_shape=%llu pool_fallback_ids_read=%llu pool_fallback_alloc=%llu pool_fallback_alloc_layer=%llu pool_fallback_alloc_pool=%llu pool_fallback_alloc_pool_data=%llu pool_fallback_alloc_shadow=%llu pool_fallback_alloc_shadow_data=%llu pool_fallback_alloc_shadow_bytes=%llu pool_fallback_plan=%llu pool_fallback_remap=%llu pool_fallback_fill=%llu pool_submit_epoch=%llu pool_completed_epoch=%llu pool_epoch_drains=%llu pool_forced_fences=%llu pool_pinned_eviction_skips=%llu delay_injections=%llu delay_us=%llu resident_cache_hits=%llu resident_cache_misses=%llu resident_cache_resets=%llu runtime_errors=%llu\n",
                (unsigned long long) state.moevm_prefetch_records,
                (unsigned long long) state.moevm_copy_prefetches,
                (unsigned long long) state.moevm_copy_prefetch_skips,
                (unsigned long long) state.moevm_copy_hook_calls,
                (unsigned long long) state.moevm_copy_hook_handled,
                (unsigned long long) state.moevm_copy_experts,
                (unsigned long long) state.moevm_copy_bytes,
                moevm_copy_mode_name(state.moevm_copy_mode_value),
                (unsigned long long) state.moevm_copy_async_ops,
                (unsigned long long) state.moevm_copy_memcpy_ops,
                (unsigned long long) state.moevm_shadow_ids_attempts,
                (unsigned long long) state.moevm_shadow_ids_applied,
                (unsigned long long) state.moevm_shadow_ids_fallback,
                (unsigned long long) state.moevm_compact_attempts,
                (unsigned long long) state.moevm_compact_applied,
                (unsigned long long) state.moevm_compact_fallback,
                (unsigned long long) state.moevm_fallback_k_overflow,
                (unsigned long long) state.moevm_pool_attempts,
                (unsigned long long) state.moevm_pool_applied,
                (unsigned long long) state.moevm_pool_fallback,
                (unsigned long long) state.moevm_pool_component_hits,
                (unsigned long long) state.moevm_pool_component_misses,
                (unsigned long long) state.moevm_pool_evictions,
                (unsigned long long) state.moevm_pool_fallback_shape,
                (unsigned long long) state.moevm_pool_fallback_ids_read,
                (unsigned long long) state.moevm_pool_fallback_alloc,
                (unsigned long long) state.moevm_pool_fallback_alloc_layer,
                (unsigned long long) state.moevm_pool_fallback_alloc_pool,
                (unsigned long long) state.moevm_pool_fallback_alloc_pool_data,
                (unsigned long long) state.moevm_pool_fallback_alloc_shadow,
                (unsigned long long) state.moevm_pool_fallback_alloc_shadow_data,
                (unsigned long long) state.moevm_pool_fallback_alloc_shadow_bytes,
                (unsigned long long) state.moevm_pool_fallback_plan,
                (unsigned long long) state.moevm_pool_fallback_remap,
                (unsigned long long) state.moevm_pool_fallback_fill,
                (unsigned long long) state.moevm_pool_submit_epoch,
                (unsigned long long) state.moevm_pool_completed_epoch,
                (unsigned long long) state.moevm_pool_epoch_drains,
                (unsigned long long) state.moevm_pool_forced_fences,
                (unsigned long long) state.moevm_pool_pinned_eviction_skips,
                (unsigned long long) state.moevm_delay_injections,
                (unsigned long long) state.moevm_delay_us,
                (unsigned long long) state.moevm_resident_cache_hits,
                (unsigned long long) state.moevm_resident_cache_misses,
                (unsigned long long) state.moevm_resident_cache_resets,
                (unsigned long long) moevm_runtime_errors);
        LOG_INF("MoEVM runtime demand_reads=%llu prefetch_reads=%llu waits=%llu wait_s=%.6f evictions=%llu high_water=%.2fGiB delay_injections=%llu delay_us=%llu\n",
                (unsigned long long) m.demand_read_ops,
                (unsigned long long) m.prefetch_read_ops,
                (unsigned long long) m.prefetch_waits,
                m.prefetch_wait_s,
                (unsigned long long) m.evictions,
                (double) m.arena_high_water_bytes / 1024.0 / 1024.0 / 1024.0,
                (unsigned long long) m.delay_injections,
                (unsigned long long) m.delay_us);
        LOG_INF("MoEVM backend copy_stats split_inputs=%llu user_inputs=%llu eligible=%llu ids_refreshes=%llu attempts=%llu handled=%llu callback_fallback=%llu graph_patches=%llu graph_patch_restores=%llu normal_fallback=%llu graph_empty=%llu non_weight=%llu non_host_weight=%llu first_node_not_mul_mat_id=%llu src0_mismatch=%llu\n",
                (unsigned long long) copy_stats.split_inputs,
                (unsigned long long) copy_stats.user_inputs,
                (unsigned long long) copy_stats.eligible,
                (unsigned long long) copy_stats.ids_refreshes,
                (unsigned long long) copy_stats.callback_attempts,
                (unsigned long long) copy_stats.callback_handled,
                (unsigned long long) copy_stats.callback_fallback,
                (unsigned long long) copy_stats.graph_patches,
                (unsigned long long) copy_stats.graph_patch_restores,
                (unsigned long long) copy_stats.normal_copy_fallback,
                (unsigned long long) copy_stats.graph_empty,
                (unsigned long long) copy_stats.non_weight_inputs,
                (unsigned long long) copy_stats.non_host_weight_inputs,
                (unsigned long long) copy_stats.first_node_not_mul_mat_id,
                (unsigned long long) copy_stats.src0_mismatch);
        LOG_INF("MoEVM backend timing backend_sync_calls=%llu backend_sync_us=%llu event_sync_calls=%llu event_sync_us=%llu event_wait_calls=%llu event_wait_us=%llu graph_compute_calls=%llu graph_compute_us=%llu user_input_wait_calls=%llu user_input_wait_us=%llu split_input_wait_calls=%llu split_input_wait_us=%llu defer_input_wait_skips=%llu defer_input_wait_disabled_events=%llu moe_input_sync_calls=%llu moe_input_sync_us=%llu ids_readback_calls=%llu ids_readback_us=%llu callback_us=%llu normal_copy_us=%llu\n",
                (unsigned long long) copy_stats.backend_sync_calls,
                (unsigned long long) copy_stats.backend_sync_us,
                (unsigned long long) copy_stats.event_sync_calls,
                (unsigned long long) copy_stats.event_sync_us,
                (unsigned long long) copy_stats.event_wait_calls,
                (unsigned long long) copy_stats.event_wait_us,
                (unsigned long long) copy_stats.graph_compute_calls,
                (unsigned long long) copy_stats.graph_compute_us,
                (unsigned long long) copy_stats.user_input_wait_calls,
                (unsigned long long) copy_stats.user_input_wait_us,
                (unsigned long long) copy_stats.split_input_wait_calls,
                (unsigned long long) copy_stats.split_input_wait_us,
                (unsigned long long) copy_stats.defer_input_wait_skips,
                (unsigned long long) copy_stats.defer_input_wait_disabled_events,
                (unsigned long long) copy_stats.moe_input_sync_calls,
                (unsigned long long) copy_stats.moe_input_sync_us,
                (unsigned long long) copy_stats.ids_readback_calls,
                (unsigned long long) copy_stats.ids_readback_us,
                (unsigned long long) copy_stats.callback_us,
                (unsigned long long) copy_stats.normal_copy_us);
        if (!runtime_error_first.empty()) {
            LOG_WRN("MoEVM runtime warning: first=%s last=%s\n", runtime_error_first.c_str(), runtime_error_last.c_str());
        }
    }
    if (state.logits_out) {
        state.close_logits_out();
        LOG_INF("logits rows written: %u\n", state.logits_rows);
    }
    if (state.out) {
        std::fclose(state.out);
    }

    if (trace_enabled) {
        LOG_INF("moe trace records written: %lld\n", (long long) state.records);
    }

    llama_backend_free();
    return ok ? 0 : 1;
}
