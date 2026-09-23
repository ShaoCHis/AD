#include "post_evaluator_flags.h"

#include <gflags/gflags.h>

DEFINE_bool(emit_dot, false, "Post-evaluator: write DOT for selected queries");
DEFINE_string(dot_dir, "dot", "Post-evaluator: directory for DOT files");
DEFINE_bool(verify_derivatives, false, "Post-evaluator: second-order central difference checks");
DEFINE_double(fd_step, 1e-5, "Post-evaluator: relative finite difference perturbation");
DEFINE_double(fd_abs_tol, 1e-6, "Post-evaluator: absolute check tolerance");
DEFINE_double(fd_rel_tol, 1e-4, "Post-evaluator: relative check tolerance");

namespace autodiff {
PostEvaluationOptions post_options_from_flags() {
    PostEvaluationOptions options;
    options.emit_dot = FLAGS_emit_dot;
    options.dot_dir = FLAGS_dot_dir;
    options.verify_derivatives = FLAGS_verify_derivatives;
    options.fd_step = FLAGS_fd_step;
    options.abs_tolerance = FLAGS_fd_abs_tol;
    options.rel_tolerance = FLAGS_fd_rel_tol;
    return options;
}
} // namespace autodiff
