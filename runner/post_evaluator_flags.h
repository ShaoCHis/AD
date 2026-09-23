#pragma once

#include "autodiff/post_evaluator.h"

namespace autodiff {
// CLI adapter: keep the post-evaluator library independent from gflags.
PostEvaluationOptions post_options_from_flags();
}
