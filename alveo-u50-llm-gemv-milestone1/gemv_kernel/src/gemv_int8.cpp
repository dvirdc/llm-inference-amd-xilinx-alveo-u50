// ============================================================================
//  gemv_int8.cpp  --  Milestone-1 INT8 GEMV kernel for Alveo U50
// ----------------------------------------------------------------------------
//  Computes y[row] = sum_{col} weights[row*cols + col] * x[col]
//      weights : int8  (signed char)  shape = rows * cols
//      x       : int8  (signed char)  length = cols
//      y       : int32                length = rows
//
//  This is the deliberately-simple "milestone 1" version:
//    * One compute unit.
//    * Scalar 8-bit loads (no 512-bit vectorization yet).
//    * 32-bit accumulator (enough headroom for any cols up to ~33M with
//      INT8 weights in [-128,127] -- proof: max |acc| <= cols * 128 * 128).
//    * Three m_axi master ports, each on its own bundle, so v++ wires them
//      to different HBM channels as configured in configs/u50_hbm.cfg.
//
//  The point of this file is *correctness*, not peak performance. Once the
//  end-to-end flow is working we will replace the inner loop with vectorized
//  loads, unrolling, multiple CUs, INT4 packing, etc. -- see TODOs at bottom.
// ============================================================================

#include <stdint.h>

// Loop tripcount upper bound used purely so the HLS reports show sensible
// latency numbers when `rows` / `cols` are runtime arguments. 4096 covers all
// of our milestone-1 test sizes (128, 512, 1024, 4096).
const int c_max_dim = 4096;

extern "C" {

void gemv_int8(
    const signed char* weights, // [rows * cols]   in  HBM
    const signed char* x,       // [cols]          in  HBM
    int*               y,       // [rows]          out HBM
    int                rows,
    int                cols
) {
    // ------------------------------------------------------------------
    // INTERFACE PRAGMAS
    // ------------------------------------------------------------------
    // m_axi: each pointer becomes an AXI4 master interface on the kernel.
    //        bundle= places it on a separate physical port so the linker
    //        can attach each port to a different HBM pseudo-channel.
    //        offset=slave: the base address is programmed via the
    //        AXI-Lite control register, which is exactly how XRT sets up
    //        BO device addresses before kernel start.
    // s_axilite: scalar arguments arrive through the AXI-Lite slave port.
    //        bundle=control puts them in the same control register block
    //        that XRT uses for start/done handshaking.
    // return on s_axilite is required for XRT to drive the kernel.
    // ------------------------------------------------------------------
#pragma HLS INTERFACE m_axi     port=weights offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=x       offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=y       offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=x       bundle=control
#pragma HLS INTERFACE s_axilite port=y       bundle=control
#pragma HLS INTERFACE s_axilite port=rows    bundle=control
#pragma HLS INTERFACE s_axilite port=cols    bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // ------------------------------------------------------------------
    // COMPUTE
    // ------------------------------------------------------------------
    // Straight nested loops. The inner loop is the dot product for one
    // output row. We pipeline the inner loop so each iteration accepts a
    // new (weight, x) pair every cycle once steady-state is reached.
    //
    // The outer loop is left as a plain loop for now -- we deliberately
    // do NOT unroll, dataflow, or shard yet. Those land in later
    // milestones; see TODOs at bottom of file.
    // ------------------------------------------------------------------
row_loop:
    for (int row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=c_max_dim

        int32_t acc = 0;

    col_loop:
        for (int col = 0; col < cols; ++col) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=c_max_dim
#pragma HLS PIPELINE II=1
            // Promote both operands to int32 before multiplying so the
            // product is computed in 32-bit and the accumulator never
            // overflows. (int8 * int8 fits in 16 bits, but we want the
            // sum to fit too.)
            int32_t w = static_cast<int32_t>(weights[row * cols + col]);
            int32_t v = static_cast<int32_t>(x[col]);
            acc += w * v;
        }

        y[row] = acc;
    }
}

} // extern "C"

// ============================================================================
//  TODOs / future milestones (do NOT implement yet -- correctness first)
// ----------------------------------------------------------------------------
//  [M2] 512-bit vectorized loads:
//       Change `weights` / `x` argument types to ap_uint<512> and unpack
//       64 int8 lanes per cycle. Requires width-converter or hls::vector.
//
//  [M2] Inner-loop unrolling:
//       #pragma HLS UNROLL factor=N on col_loop after vectorizing, so
//       N int8 MACs happen per cycle.
//
//  [M3] HBM sharding:
//       Split `weights` across HBM[0..7] using `sp=...:HBM[0:7]` and
//       parallel m_axi ports gmem0..gmem7. Each port feeds its own
//       dot-product lane.
//
//  [M3] Multiple compute units (nk=gemv_int8:N:gemv_int8_1.gemv_int8_2...):
//       Run N CUs in parallel, each owning a row-tile of the weight
//       matrix. Host dispatches N kernel runs concurrently.
//
//  [M4] INT4 packed weights:
//       Change `weights` to const ap_uint<4>* or pack 2x int4 per byte.
//       Double effective memory bandwidth.
//
//  [M5..] RMSNorm fusion, Q/K/V projection, MLP, KV cache attention.
// ============================================================================
