# TODO(herhut): describe this package.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

load(
    "//tensorflow/core/platform:default/cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

tf_cc_test(
    name = "mlir_gpu_lhlo_gen_test",
    srcs = if_cuda_is_configured(["mlir_gpu_lhlo_gen_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/service:mlir_gpu_plugin",
        "//tensorflow/compiler/xla/service/mlir_gpu:mlir_irgen_test_base",
        "//tensorflow/stream_executor/lib",
    ]),
)
