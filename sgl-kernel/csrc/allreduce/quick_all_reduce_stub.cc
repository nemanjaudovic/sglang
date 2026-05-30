// Stub implementations for quick_all_reduce on RDNA GPUs.
// Quick allreduce uses CDNA-specific buffer resource instructions and is
// only needed for multi-GPU allreduce, which RDNA desktop GPUs don't use.

#include <torch/all.h>

#include <optional>
#include <stdexcept>
#include <vector>

using fptr_t = int64_t;

fptr_t init_custom_qr(int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size) {
  throw std::runtime_error("quick_all_reduce is not supported on RDNA GPUs");
}

void qr_destroy(fptr_t _fa) {}

torch::Tensor qr_get_handle(fptr_t _fa) {
  throw std::runtime_error("quick_all_reduce is not supported on RDNA GPUs");
}

void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles) {
  throw std::runtime_error("quick_all_reduce is not supported on RDNA GPUs");
}

void qr_all_reduce(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, int64_t quant_level, bool cast_bf2half) {
  throw std::runtime_error("quick_all_reduce is not supported on RDNA GPUs");
}

int64_t qr_max_size() { return 0; }
