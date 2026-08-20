#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator &matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    size_t r = i + 1;
    auto current_query = rater.GetNextQuery();

    // Phase 1: Move keys/values/query to SRAM first (batch IO to allow parallelism)
    for (size_t j = 0; j < r; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);
    }
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Phase 2: Build K and V in SRAM (much faster than HBM)
    Matrix* K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
    if (r == 1) {
      gpu_sim.Copy(keys[0], K, kInSharedMemory);
    } else {
      Matrix* prev = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(i) + "_0");
      gpu_sim.Concat(keys[0], keys[1], prev, 0, kInSharedMemory);
      for (size_t j = 2; j < r; ++j) {
        Matrix* new_prev = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(prev, keys[j], new_prev, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(prev);
        prev = new_prev;
      }
      gpu_sim.Copy(prev, K, kInSharedMemory);
      gpu_sim.ReleaseMatrix(prev);
    }

    Matrix* V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
    if (r == 1) {
      gpu_sim.Copy(values[0], V, kInSharedMemory);
    } else {
      Matrix* prev_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i) + "_0");
      gpu_sim.Concat(values[0], values[1], prev_v, 0, kInSharedMemory);
      for (size_t j = 2; j < r; ++j) {
        Matrix* new_prev_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(prev_v, values[j], new_prev_v, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(prev_v);
        prev_v = new_prev_v;
      }
      gpu_sim.Copy(prev_v, V, kInSharedMemory);
      gpu_sim.ReleaseMatrix(prev_v);
    }

    // Transpose K in SRAM (much faster than HBM)
    gpu_sim.Transpose(K, kInSharedMemory);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Phase 3: Compute logits = Q @ K^T
    Matrix* logits = matrix_memory_allocator.Allocate("logits_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K, logits);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.ReleaseMatrix(K);

    // Phase 4: Process each query row individually to avoid large 2nd MatMul
    std::vector<Matrix*> output_rows;
    for (size_t q_idx = 0; q_idx < r; ++q_idx) {
      // Get current logits row
      Matrix* logits_row = matrix_memory_allocator.Allocate("logits_row_" + std::to_string(i) + "_" + std::to_string(q_idx));
      gpu_sim.GetRow(logits, q_idx, logits_row, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);

      // Compute softmax for this row
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(q_idx));
      gpu_sim.MatExp(logits_row, exp_row);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(logits_row);

      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(i) + "_" + std::to_string(q_idx));
      gpu_sim.Sum(exp_row, sum_exp);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(q_idx));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);

      // Compute output row = sum over j (softmax[j] * V[j])
      Matrix* output_row = matrix_memory_allocator.Allocate("output_row_" + std::to_string(i) + "_" + std::to_string(q_idx));
      // Initialize output_row to zeros
      // First row: use weighted V[0]
      Matrix* s0 = matrix_memory_allocator.Allocate("s0_" + std::to_string(i) + "_" + std::to_string(q_idx));
      gpu_sim.GetColumn(softmax_row, 0, s0, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);

      Matrix* w0 = matrix_memory_allocator.Allocate("w0_" + std::to_string(i) + "_" + std::to_string(q_idx));
      gpu_sim.MatMulNum(values[0], s0, w0);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(s0);

      // Copy w0 to output_row
      gpu_sim.Copy(w0, output_row, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(w0);

      // Add remaining V[j] weighted by softmax[j]
      for (size_t j = 1; j < r; ++j) {
        Matrix* sj = matrix_memory_allocator.Allocate("sj_" + std::to_string(i) + "_" + std::to_string(q_idx) + "_" + std::to_string(j));
        gpu_sim.GetColumn(softmax_row, j, sj, kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);

        Matrix* wj = matrix_memory_allocator.Allocate("wj_" + std::to_string(i) + "_" + std::to_string(q_idx) + "_" + std::to_string(j));
        gpu_sim.MatMulNum(values[j], sj, wj);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(sj);

        Matrix* new_output_row = matrix_memory_allocator.Allocate("new_output_row_" + std::to_string(i) + "_" + std::to_string(q_idx) + "_" + std::to_string(j));
        gpu_sim.MatAdd(output_row, wj, new_output_row);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(output_row);
        gpu_sim.ReleaseMatrix(wj);
        output_row = new_output_row;
      }
      gpu_sim.ReleaseMatrix(softmax_row);

      output_rows.push_back(output_row);
    }
    gpu_sim.ReleaseMatrix(logits);
    gpu_sim.ReleaseMatrix(V);

    // Phase 5: Concatenate output rows to get final output
    Matrix* output = matrix_memory_allocator.Allocate("output_" + std::to_string(i));
    if (r == 1) {
      gpu_sim.Copy(output_rows[0], output, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(output_rows[0]);
    } else {
      Matrix* temp_output = matrix_memory_allocator.Allocate("temp_output_" + std::to_string(i) + "_0");
      gpu_sim.Concat(output_rows[0], output_rows[1], temp_output, 0, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(output_rows[0]);
      gpu_sim.ReleaseMatrix(output_rows[1]);

      for (size_t j = 2; j < r; ++j) {
        Matrix* new_temp_output = matrix_memory_allocator.Allocate("temp_output_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(temp_output, output_rows[j], new_temp_output, 0, kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(temp_output);
        gpu_sim.ReleaseMatrix(output_rows[j]);
        temp_output = new_temp_output;
      }

      gpu_sim.Copy(temp_output, output, kInSharedMemory);
      gpu_sim.Run(false, &matrix_memory_allocator);
      gpu_sim.ReleaseMatrix(temp_output);
    }

    // Phase 6: Move output to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(output);
    gpu_sim.Run(false, &matrix_memory_allocator);

    rater.CommitAnswer(*output);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
