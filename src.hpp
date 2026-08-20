#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    size_t r = i + 1;
    auto current_query = rater.GetNextQuery();

    // Phase 1: Build K and V in HBM, transpose K
    Matrix* K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
    if (r == 1) {
        gpu_sim.Copy(keys[0], K, kInGpuHbm);
    } else {
        Matrix* prev = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(i) + "_0");
        gpu_sim.Concat(keys[0], keys[1], prev, 0, kInGpuHbm);
        for (size_t j = 2; j < r; ++j) {
            Matrix* new_prev = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(i) + "_" + std::to_string(j));
            gpu_sim.Concat(prev, keys[j], new_prev, 0, kInGpuHbm);
            gpu_sim.ReleaseMatrix(prev);
            prev = new_prev;
        }
        gpu_sim.Copy(prev, K, kInGpuHbm);
        gpu_sim.ReleaseMatrix(prev);
    }

    Matrix* V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
    if (r == 1) {
        gpu_sim.Copy(values[0], V, kInGpuHbm);
    } else {
        Matrix* prev_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i) + "_0");
        gpu_sim.Concat(values[0], values[1], prev_v, 0, kInGpuHbm);
        for (size_t j = 2; j < r; ++j) {
            Matrix* new_prev_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i) + "_" + std::to_string(j));
            gpu_sim.Concat(prev_v, values[j], new_prev_v, 0, kInGpuHbm);
            gpu_sim.ReleaseMatrix(prev_v);
            prev_v = new_prev_v;
        }
        gpu_sim.Copy(prev_v, V, kInGpuHbm);
        gpu_sim.ReleaseMatrix(prev_v);
    }

    gpu_sim.Transpose(K, kInGpuHbm);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Phase 2: Move Q, K, V to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K);
    gpu_sim.MoveMatrixToSharedMem(V);
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Phase 3: Compute logits, softmax, attention weights, output in SRAM
    Matrix* logits = matrix_memory_allocator.Allocate("logits_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K, logits);
    gpu_sim.Run(false, &matrix_memory_allocator);

    std::vector<Matrix*> softmax_rows;
    for (size_t j = 0; j < r; ++j) {
        Matrix* row_j = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.GetRow(logits, j, row_j, kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);

        Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.MatExp(row_j, exp_row);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(row_j);

        Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Sum(exp_row, sum_exp);
        gpu_sim.Run(false, &matrix_memory_allocator);

        Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(exp_row);
        gpu_sim.ReleaseMatrix(sum_exp);

        softmax_rows.push_back(softmax_row);
    }
    gpu_sim.ReleaseMatrix(logits);

    Matrix* attention_weights = matrix_memory_allocator.Allocate("attention_weights_" + std::to_string(i));
    if (r == 1) {
        gpu_sim.Copy(softmax_rows[0], attention_weights, kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(softmax_rows[0]);
    } else {
        Matrix* temp_soft = matrix_memory_allocator.Allocate("temp_soft_" + std::to_string(i) + "_0");
        gpu_sim.Concat(softmax_rows[0], softmax_rows[1], temp_soft, 0, kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(softmax_rows[0]);
        gpu_sim.ReleaseMatrix(softmax_rows[1]);

        for (size_t j = 2; j < r; ++j) {
            Matrix* new_temp_soft = matrix_memory_allocator.Allocate("temp_soft_" + std::to_string(i) + "_" + std::to_string(j));
            gpu_sim.Concat(temp_soft, softmax_rows[j], new_temp_soft, 0, kInSharedMemory);
            gpu_sim.Run(false, &matrix_memory_allocator);
            gpu_sim.ReleaseMatrix(temp_soft);
            gpu_sim.ReleaseMatrix(softmax_rows[j]);
            temp_soft = new_temp_soft;
        }

        gpu_sim.Copy(temp_soft, attention_weights, kInSharedMemory);
        gpu_sim.Run(false, &matrix_memory_allocator);
        gpu_sim.ReleaseMatrix(temp_soft);
    }

    Matrix* output = matrix_memory_allocator.Allocate("output_" + std::to_string(i));
    gpu_sim.MatMul(attention_weights, V, output);
    gpu_sim.Run(false, &matrix_memory_allocator);
    gpu_sim.ReleaseMatrix(attention_weights);
    gpu_sim.ReleaseMatrix(K);
    gpu_sim.ReleaseMatrix(V);

    // Phase 4: Move output to HBM and commit
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
