# YuNet int8bq: compact, fast C for RV64 / x86

Model: `face_detection_yunet_2023mar_int8bq.onnx` (OpenCV Zoo, weight-only block-quantized INT8; included).

- `yunet_gen.py` + `runtime.c`: generate `yunet_fast.c` (`python yunet_gen.py model.onnx yunet_fast.c`). Same `entry()` signature as onnx2c output.
- `yunet_fast.c`: generated output (~100 KB vs ~500 KB from onnx2c). INT8 weights are kept in a base64 string and dequantized once on first call.
- `ref.py` / `harness.c` / `cmp.py`: reference check. `ref.py` folds `DequantizeLinear` in numpy (onnxruntime rejects 1-D blocked dequantize) and writes `input.bin`/`ref.bin`; `harness.c` runs `entry()`; `cmp.py` compares.

Build:

    gcc -O3 -march=native -fopenmp harness.c yunet_fast.c -lm
    riscv64-linux-gnu-gcc -O3 -march=rv64gcv -fopenmp -static harness.c yunet_fast.c -lm   # RVV intrinsics

OpenMP threads are capped at 8 by default (`-DYUNET_THREADS=n` to change). The first call initializes weights and is not thread-safe.
Verified: max abs error ~5e-6 vs reference on x86 and on RV64 under QEMU (VLEN 128/256/512). Not measured on real RV64 hardware.
