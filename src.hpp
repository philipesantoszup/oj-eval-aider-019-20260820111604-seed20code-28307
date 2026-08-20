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

    // Step 1: Concatenate first r keys into K (shape [r, d])
    Matrix* K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
    if (r == 1) {
        gpu_sim.Copy(keys[0], K, kInGpuHbm);
    } else {
        Matrix* temp = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(i) + "_0");
        gpu_sim.Concat(keys[0], keys[1], temp, 0, kInGpuHbm);
        for (size_t j = 2; j < r; ++j) {
            Matrix* new_temp = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(i) + "_" + std::to_string(j));
            gpu_sim.Concat(temp, keys[j], new_temp, 0, kInGpuHbm);
            gpu_sim.ReleaseMatrix(temp);
            temp = new_temp;
        }
        gpu_sim.Copy(temp, K, kInGpuHbm);
        gpu_sim.ReleaseMatrix(temp);
    }

    // Step 2: Concatenate first r values into V (shape [r, d])
    Matrix* V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
    if (r == 1) {
        gpu_sim.Copy(values[0], V, kInGpuHbm);
    } else {
        Matrix* temp_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i) + "_0");
        gpu_sim.Concat(values[0], values[1], temp_v, 0, kInGpuHbm);
        for (size_t j = 2; j < r; ++j) {
            Matrix* new_temp_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i) + "_" + std::to_string(j));
            gpu_sim.Concat(temp_v, values[j], new_temp_v, 0, kInGpuHbm);
            gpu_sim.ReleaseMatrix(temp_v);
            temp_v = new_temp_v;
        }
        gpu_sim.Copy(temp_v, V, kInGpuHbm);
        gpu_sim.ReleaseMatrix(temp_v);
    }

    // Step 3: Compute K^T
    Matrix* K_T = matrix_memory_allocator.Allocate("K_T_" + std::to_string(i));
    gpu_sim.Copy(K, K_T, kInGpuHbm);
    gpu_sim.Transpose(K_T, kInGpuHbm);
    // K is no longer needed after this point, release it
    gpu_sim.ReleaseMatrix(K);

    // Step 4: Move Q, K_T, V to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_T);
    gpu_sim.MoveMatrixToSharedMem(V);

    // Step 5: Compute logits = Q @ K^T
    Matrix* logits = matrix_memory_allocator.Allocate("logits_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_T, logits);
    // K_T is no longer needed after this point
    gpu_sim.ReleaseMatrix(K_T);

    // Step 6: Compute row-wise Softmax to get attention_weights
    std::vector<Matrix*> softmax_rows;
    for (size_t j = 0; j < r; ++j) {
        Matrix* row_j = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.GetRow(logits, j, row_j, kInSharedMemory);

        Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.MatExp(row_j, exp_row);
        gpu_sim.ReleaseMatrix(row_j);

        Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Sum(exp_row, sum_exp);

        Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
        gpu_sim.ReleaseMatrix(exp_row);
        gpu_sim.ReleaseMatrix(sum_exp);

        softmax_rows.push_back(softmax_row);
    }
    // logits is no longer needed
    gpu_sim.ReleaseMatrix(logits);

    // Step 7: Concatenate softmax rows into attention_weights
    Matrix* attention_weights = matrix_memory_allocator.Allocate("attention_weights_" + std::to_string(i));
    if (r == 1) {
        gpu_sim.Copy(softmax_rows[0], attention_weights, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_rows[0]);
    } else {
        Matrix* temp_soft = matrix_memory_allocator.Allocate("temp_soft_" + std::to_string(i) + "_0");
        gpu_sim.Concat(softmax_rows[0], softmax_rows[1], temp_soft, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_rows[0]);
        gpu_sim.ReleaseMatrix(softmax_rows[1]);
        for (size_t j = 2; j < r; ++j) {
            Matrix* new_temp_soft = matrix_memory_allocator.Allocate("temp_soft_" + std::to_string(i) + "_" + std::to_string(j));
            gpu_sim.Concat(temp_soft, softmax_rows[j], new_temp_soft, 0, kInSharedMemory);
            gpu_sim.ReleaseMatrix(temp_soft);
            gpu_sim.ReleaseMatrix(softmax_rows[j]);
            temp_soft = new_temp_soft;
        }
        gpu_sim.Copy(temp_soft, attention_weights, kInSharedMemory);
        gpu_sim.ReleaseMatrix(temp_soft);
    }

    // Step 8: Compute output = attention_weights @ V
    Matrix* output = matrix_memory_allocator.Allocate("output_" + std::to_string(i));
    gpu_sim.MatMul(attention_weights, V, output);
    // attention_weights and V are no longer needed
    gpu_sim.ReleaseMatrix(attention_weights);
    gpu_sim.ReleaseMatrix(V);

    // Step 9: Move output to HBM
    gpu_sim.MoveMatrixToGpuHbm(output);

    // Step 10: Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Step 11: Commit the answer
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
