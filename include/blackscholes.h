#ifndef BLACKSCHOLES_H
#define BLACKSCHOLES_H

#include "bench.h"
#include <cmath>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_BLACKSCHOLES{4800000};

struct blackscholes_args {
    std::vector<float> call_option_price;
    std::vector<float> put_option_price;
    std::vector<float> spot_price;
    std::vector<float> strike;
    std::vector<float> rate;
    std::vector<float> volatility;
    std::vector<float> time;
    double epsilon = 5e-3;
    // TODO: You may want to add new params at the end...
};

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n = 81920,
                             std::uint32_t seed = 12345u);

/* Cumulative Normal Distribution Function */
void CNDF(float &InputX, float &OutputX);

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time);
// TODO: Implement your version for BlkSchls, then call it in stu_BlkSchls_wrapper
void stu_BlkSchls(std::vector<float> &CallOptionPrice,
                  std::vector<float> &PutOptionPrice,
                  const std::vector<float> &spotPrice,
                  const std::vector<float> &strike,
                  const std::vector<float> &rate,
                  const std::vector<float> &volatility,
                  const std::vector<float> &time);

void naive_BlkSchls_wrapper(void *ctx);
void stu_BlkSchls_wrapper(void *ctx);

bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func);

#endif
