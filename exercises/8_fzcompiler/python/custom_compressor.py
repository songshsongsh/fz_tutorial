#!/usr/bin/env python
import libpressio as lp
import numpy as np
from pprint import pprint
from pathlib import Path

SOURCE = r"""
#include <boost/config.hpp>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cmath>

#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

#include <torch/extension.h>
#include <omp.h>

namespace libpressio { namespace compressors { namespace fzdemo_ns {

namespace {

static std::vector<int64_t> to_int64_dims(pressio_data const& data) {
    std::vector<int64_t> dims;
    dims.reserve(data.dimensions().size());
    for (auto d : data.dimensions()) {
        dims.push_back(static_cast<int64_t>(d));
    }
    return dims;
}

static torch::Tensor pressio_to_tensor_f32(pressio_data const& data) {
    if (data.dtype() != pressio_float_dtype) {
        throw std::runtime_error("pressio_to_tensor_f32 only supports pressio_float_dtype");
    }

    auto const& pdims = data.dimensions();
    auto* ptr = static_cast<float*>(data.data());

    if (pdims.size() != 2) {
        throw std::runtime_error("expected 2D float input");
    }

    return torch::from_blob(
        ptr,
        {
            static_cast<int64_t>(pdims[1]),
            static_cast<int64_t>(pdims[0])
        },
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
}

static pressio_data pack_compressed_tensor(torch::Tensor const& compressed) {
    auto c = compressed.contiguous();

    if (c.scalar_type() != torch::kInt32) {
        throw std::runtime_error("compressed tensor must be int32");
    }
    if (c.dim() != 1) {
        throw std::runtime_error("compressed tensor must be 1D");
    }

    const size_t payload_len = c.numel();

    auto out = pressio_data::owning(pressio_int32_dtype, {payload_len});
    auto* dst = static_cast<int32_t*>(out.data());

    std::memcpy(
        dst,
        c.data_ptr<int32_t>(),
        payload_len * sizeof(int32_t)
    );

    return out;
}

struct unpacked_compressed {
    torch::Tensor payload;
};

static unpacked_compressed unpack_compressed_tensor(pressio_data const& data) {
    if (data.dtype() != pressio_int32_dtype) {
        throw std::runtime_error("compressed pressio_data must be int32");
    }
    if (data.dimensions().size() != 1) {
        throw std::runtime_error("compressed pressio_data must be 1D");
    }

    pressio_data readable = domain_manager().make_readable(
        domain_plugins().build("malloc"),
        data
    );

    auto* src = static_cast<int32_t*>(readable.data());
    const size_t total_len = readable.num_elements();

    auto payload = torch::from_blob(
        src,
        {static_cast<int64_t>(total_len)},
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();

    return {payload};
}
static pressio_data tensor_to_pressio_f32(torch::Tensor const& tensor) {
    auto t = tensor.contiguous();

    if (t.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("restored tensor must be float32");
    }

    std::vector<size_t> dims;
    dims.reserve(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        dims.push_back(static_cast<size_t>(t.size(i)));
    }

    auto out = pressio_data::owning(pressio_float_dtype, dims);
    std::memcpy(
        out.data(),
        t.data_ptr<float>(),
        t.numel() * sizeof(float)
    );

    return out;
}

} // anonymous namespace


// ===========================
// generated compression code
// ===========================
torch::Tensor concat_concat_ceil_log2_lufqwjkb(int num_blocks, int block_size, torch::Tensor obj_input, float eb)
{
    auto input = obj_input.accessor<float, 2>();
torch::Tensor obj_arr5 = torch::empty({num_blocks,block_size}, at::kFloat);
auto arr5 = obj_arr5.accessor<float, 2>();
#pragma omp parallel for num_threads(16)
for (int _l0 = 0; _l0 < num_blocks; _l0 += 1) {
for (int _l1 = 0; _l1 < block_size; _l1 += 1) {
arr5[_l0][_l1] = (input[_l0][_l1] * (0.5 / eb));
}
}
torch::Tensor obj_arr13 = torch::empty({num_blocks,block_size}, at::kBool);
auto arr13 = obj_arr13.accessor<bool, 2>();
#pragma omp parallel for num_threads(16)
for (int _l2 = 0; _l2 < num_blocks; _l2 += 1) {
for (int _l3 = 0; _l3 < block_size; _l3 += 1) {
arr13[_l2][_l3] = (arr5[_l2][_l3] > -0.5);
}
}
torch::Tensor obj_arr21 = torch::empty({num_blocks,block_size}, at::kFloat);
auto arr21 = obj_arr21.accessor<float, 2>();
#pragma omp parallel for num_threads(16)
for (int _l4 = 0; _l4 < num_blocks; _l4 += 1) {
for (int _l5 = 0; _l5 < block_size; _l5 += 1) {
arr21[_l4][_l5] = (arr5[_l4][_l5] + 0.5);
}
}
torch::Tensor obj_arr29 = torch::empty({num_blocks,block_size}, at::kFloat);
auto arr29 = obj_arr29.accessor<float, 2>();
#pragma omp parallel for num_threads(16)
for (int _l6 = 0; _l6 < num_blocks; _l6 += 1) {
for (int _l7 = 0; _l7 < block_size; _l7 += 1) {
arr29[_l6][_l7] = (arr5[_l6][_l7] - 0.5);
}
}
torch::Tensor obj_arr36 = torch::empty({num_blocks,block_size}, at::kFloat);
auto arr36 = obj_arr36.accessor<float, 2>();
for (int _l8 = 0; _l8 < num_blocks; _l8 += 1) {
for (int _l9 = 0; _l9 < block_size; _l9 += 1) {
if (arr13[_l8][_l9]) {
arr36[_l8][_l9] = arr21[_l8][_l9];
}
else {
arr36[_l8][_l9] = arr29[_l8][_l9];
}
}
}
torch::Tensor obj_arr47 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr47 = obj_arr47.accessor<int, 2>();
for (int _l10 = 0; _l10 < num_blocks; _l10 += 1) {
for (int _l11 = 0; _l11 < block_size; _l11 += 1) {
arr47[_l10][_l11] = static_cast<int>(arr36[_l10][_l11]);
}
}
torch::Tensor obj_arr54 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr54 = obj_arr54.accessor<int, 2>();
for (int _l12 = 0; _l12 < num_blocks; _l12 += 1) {
arr54[_l12][0] = arr47[_l12][0];
for (int _l13 = 1; _l13 < block_size; _l13 += 1) {
arr54[_l12][_l13] = (arr47[_l12][_l13] - arr47[_l12][(_l13 - 1)]);
}
}
torch::Tensor obj_arr69 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr69 = obj_arr69.accessor<int, 2>();
#pragma omp parallel for num_threads(16)
for (int _l14 = 0; _l14 < num_blocks; _l14 += 1) {
for (int _l15 = 0; _l15 < block_size; _l15 += 1) {
arr69[_l14][_l15] = abs(arr54[_l14][_l15]);
}
}
torch::Tensor obj_arr77 = torch::empty({num_blocks}, at::kInt);
auto arr77 = obj_arr77.accessor<int, 1>();
for (int _l18 = 0; _l18 < num_blocks; _l18 += 1) {
arr77[_l18] = -2147483648;
for (int _l17 = 0; _l17 < block_size; _l17 += 1) {
arr77[_l18] = (arr77[_l18] > arr69[_l18][_l17] ? (arr77[_l18]) : (arr69[_l18][_l17]));
}
}
torch::Tensor obj_arr90 = torch::empty({num_blocks}, at::kInt);
auto arr90 = obj_arr90.accessor<int, 1>();
#pragma omp parallel for num_threads(16)
for (int _l19 = 0; _l19 < num_blocks; _l19 += 1) {
arr90[_l19] = (arr77[_l19] + 1);
}
torch::Tensor obj_arr94 = torch::empty({num_blocks}, at::kFloat);
auto arr94 = obj_arr94.accessor<float, 1>();
#pragma omp parallel for num_threads(16)
for (int _l20 = 0; _l20 < num_blocks; _l20 += 1) {
arr94[_l20] = log2(arr90[_l20]);
}
torch::Tensor obj_arr98 = torch::empty({num_blocks}, at::kInt);
auto arr98 = obj_arr98.accessor<int, 1>();
#pragma omp parallel for num_threads(16)
for (int _l21 = 0; _l21 < num_blocks; _l21 += 1) {
arr98[_l21] = ceil(arr94[_l21]);
}
torch::Tensor obj_arr118 = torch::empty({num_blocks}, at::kBool);
auto arr118 = obj_arr118.accessor<bool, 1>();
#pragma omp parallel for num_threads(16)
for (int _l26 = 0; _l26 < num_blocks; _l26 += 1) {
arr118[_l26] = (arr98[_l26] > 0);
}
torch::Tensor obj_arr122 = torch::empty({num_blocks}, at::kInt);
auto arr122 = obj_arr122.accessor<int, 1>();
for (int _l27 = 0; _l27 < num_blocks; _l27 += 1) {
arr122[_l27] = static_cast<int>(arr118[_l27]);
}
int s159;
s159 = 0;
for (int _l31 = 0; _l31 < num_blocks; _l31 += 1) {
s159 = (s159 + arr122[_l31]);
}
int s209;
s209 = 0;
for (int _l38 = 0; _l38 < num_blocks; _l38 += 1) {
s209 = (s209 + arr98[_l38]);
}
torch::Tensor obj_arr103 = torch::empty({num_blocks,block_size}, at::kBool);
auto arr103 = obj_arr103.accessor<bool, 2>();
#pragma omp parallel for num_threads(16)
for (int _l22 = 0; _l22 < num_blocks; _l22 += 1) {
for (int _l23 = 0; _l23 < block_size; _l23 += 1) {
arr103[_l22][_l23] = (arr54[_l22][_l23] > 0);
}
}
torch::Tensor obj_arr110 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr110 = obj_arr110.accessor<int, 2>();
for (int _l24 = 0; _l24 < num_blocks; _l24 += 1) {
for (int _l25 = 0; _l25 < block_size; _l25 += 1) {
arr110[_l24][_l25] = static_cast<int>(arr103[_l24][_l25]);
}
}
torch::Tensor obj_arr126 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr126 = obj_arr126.accessor<int, 2>();
uint64_t bits_126;
int bitcnt_126;
int outidx_126;
uint64_t mask_126;
bits_126 = static_cast<uint64_t>(0);
bitcnt_126 = 0;
outidx_126 = 0;
for (int _l28 = 0; _l28 < num_blocks; _l28 += 1) {
for (int _l29 = 0; _l29 < block_size; _l29 += 1) {
if ((arr122[_l28] == 0)) {
mask_126 = static_cast<uint64_t>(0);
}
else {
if ((arr122[_l28] == 32)) {
mask_126 = static_cast<uint64_t>(4294967295);
}
else {
mask_126 = ((static_cast<uint64_t>(1) << arr122[_l28]) - static_cast<uint64_t>(1));
}
}
bits_126 = (bits_126 | ((static_cast<uint64_t>(arr110[_l28][_l29]) & mask_126) << bitcnt_126));
bitcnt_126 = (bitcnt_126 + arr122[_l28]);
if ((bitcnt_126 >= 32)) {
arr126[(outidx_126 / block_size)][(outidx_126 % block_size)] = static_cast<int>((bits_126 & static_cast<uint64_t>(4294967295)));
outidx_126 = (outidx_126 + 1);
bitcnt_126 = (bitcnt_126 - 32);
bits_126 = (bits_126 >> 32);
}
else {
}
}
}
if ((bitcnt_126 > 0)) {
arr126[(outidx_126 / block_size)][(outidx_126 % block_size)] = static_cast<int>((bits_126 & static_cast<uint64_t>(4294967295)));
outidx_126 = (outidx_126 + 1);
}
else {
}
for (int _l30 = outidx_126; _l30 < (num_blocks * block_size); _l30 += 1) {
arr126[(_l30 / block_size)][(_l30 % block_size)] = 0;
}
torch::Tensor obj_arr163 = torch::empty({s159}, at::kInt);
auto arr163 = obj_arr163.accessor<int, 1>();
for (int _l33 = 0; _l33 < s159; _l33 += 1) {
arr163[_l33] = arr126[(_l33 / block_size)][(_l33 % block_size)];
}
torch::Tensor obj_arr169 = torch::empty({(num_blocks + s159)}, at::kInt);
auto arr169 = obj_arr169.accessor<int, 1>();
for (int _l32 = 0; _l32 < num_blocks; _l32 += 1) {
arr169[_l32] = arr98[_l32];
}
for (int _l34 = 0; _l34 < s159; _l34 += 1) {
arr169[(_l34 + num_blocks)] = arr163[_l34];
}
torch::Tensor obj_arr176 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr176 = obj_arr176.accessor<int, 2>();
uint64_t bits_176;
int bitcnt_176;
int outidx_176;
uint64_t mask_176;
bits_176 = static_cast<uint64_t>(0);
bitcnt_176 = 0;
outidx_176 = 0;
for (int _l35 = 0; _l35 < num_blocks; _l35 += 1) {
for (int _l36 = 0; _l36 < block_size; _l36 += 1) {
if ((arr98[_l35] == 0)) {
mask_176 = static_cast<uint64_t>(0);
}
else {
if ((arr98[_l35] == 32)) {
mask_176 = static_cast<uint64_t>(4294967295);
}
else {
mask_176 = ((static_cast<uint64_t>(1) << arr98[_l35]) - static_cast<uint64_t>(1));
}
}
bits_176 = (bits_176 | ((static_cast<uint64_t>(arr69[_l35][_l36]) & mask_176) << bitcnt_176));
bitcnt_176 = (bitcnt_176 + arr98[_l35]);
if ((bitcnt_176 >= 32)) {
arr176[(outidx_176 / block_size)][(outidx_176 % block_size)] = static_cast<int>((bits_176 & static_cast<uint64_t>(4294967295)));
outidx_176 = (outidx_176 + 1);
bitcnt_176 = (bitcnt_176 - 32);
bits_176 = (bits_176 >> 32);
}
else {
}
}
}
if ((bitcnt_176 > 0)) {
arr176[(outidx_176 / block_size)][(outidx_176 % block_size)] = static_cast<int>((bits_176 & static_cast<uint64_t>(4294967295)));
outidx_176 = (outidx_176 + 1);
}
else {
}
for (int _l37 = outidx_176; _l37 < (num_blocks * block_size); _l37 += 1) {
arr176[(_l37 / block_size)][(_l37 % block_size)] = 0;
}
torch::Tensor obj_arr213 = torch::empty({s209}, at::kInt);
auto arr213 = obj_arr213.accessor<int, 1>();
for (int _l39 = 0; _l39 < s209; _l39 += 1) {
arr213[_l39] = arr176[(_l39 / block_size)][(_l39 % block_size)];
}
torch::Tensor obj_arr219 = torch::empty({((num_blocks + s159) + s209)}, at::kInt);
auto arr219 = obj_arr219.accessor<int, 1>();
for (int _l40 = 0; _l40 < (num_blocks + s159); _l40 += 1) {
arr219[_l40] = arr169[_l40];
}
for (int _l41 = 0; _l41 < s209; _l41 += 1) {
arr219[(_l41 + (num_blocks + s159))] = arr213[_l41];
}
return obj_arr219;
}


// =============================
// generated decompression code
// =============================
torch::Tensor mul_cast_prefix_sum_maskiynoonyk(int num_blocks, int block_size, int input_size, torch::Tensor obj_input, float eb)
{
    auto input = obj_input.accessor<int, 1>();
torch::Tensor obj_arr7 = torch::empty({num_blocks}, at::kInt);
auto arr7 = obj_arr7.accessor<int, 1>();
for (int _l1 = 0; _l1 < num_blocks; _l1 += 1) {
arr7[_l1] = input[_l1];
}
torch::Tensor obj_arr12 = torch::empty({num_blocks}, at::kBool);
auto arr12 = obj_arr12.accessor<bool, 1>();
#pragma omp parallel for num_threads(16)
for (int _l2 = 0; _l2 < num_blocks; _l2 += 1) {
arr12[_l2] = (arr7[_l2] > 0);
}
torch::Tensor obj_arr16 = torch::empty({num_blocks}, at::kInt);
auto arr16 = obj_arr16.accessor<int, 1>();
for (int _l3 = 0; _l3 < num_blocks; _l3 += 1) {
arr16[_l3] = static_cast<int>(arr12[_l3]);
}
int s21;
s21 = 0;
for (int _l4 = 0; _l4 < num_blocks; _l4 += 1) {
s21 = (s21 + arr16[_l4]);
}
torch::Tensor obj_arr3 = torch::empty({(input_size - num_blocks)}, at::kInt);
auto arr3 = obj_arr3.accessor<int, 1>();
for (int _l0 = 0; _l0 < (input_size - num_blocks); _l0 += 1) {
arr3[_l0] = input[(_l0 + num_blocks)];
}
torch::Tensor obj_arr25 = torch::empty({s21}, at::kInt);
auto arr25 = obj_arr25.accessor<int, 1>();
for (int _l5 = 0; _l5 < s21; _l5 += 1) {
arr25[_l5] = arr3[_l5];
}
torch::Tensor obj_arr30 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr30 = obj_arr30.accessor<int, 2>();
uint64_t ubits_30;
int ubitcnt_30;
int uinidx_30;
uint64_t umask_30;
ubits_30 = static_cast<uint64_t>(0);
ubitcnt_30 = 0;
uinidx_30 = 0;
for (int _l6 = 0; _l6 < num_blocks; _l6 += 1) {
for (int _l7 = 0; _l7 < block_size; _l7 += 1) {
if ((arr16[_l6] == 0)) {
arr30[_l6][_l7] = 0;
}
else {
if ((ubitcnt_30 < arr16[_l6])) {
ubits_30 = (ubits_30 | (static_cast<uint64_t>((arr25[uinidx_30] & 4294967295)) << ubitcnt_30));
ubitcnt_30 = (ubitcnt_30 + 32);
uinidx_30 = (uinidx_30 + 1);
}
else {
}
if ((ubitcnt_30 < arr16[_l6])) {
ubits_30 = (ubits_30 | (static_cast<uint64_t>((arr25[uinidx_30] & 4294967295)) << ubitcnt_30));
ubitcnt_30 = (ubitcnt_30 + 32);
uinidx_30 = (uinidx_30 + 1);
}
else {
}
if ((arr16[_l6] == 32)) {
umask_30 = static_cast<uint64_t>(4294967295);
}
else {
umask_30 = ((static_cast<uint64_t>(1) << arr16[_l6]) - static_cast<uint64_t>(1));
}
arr30[_l6][_l7] = static_cast<int>((ubits_30 & umask_30));
ubits_30 = (ubits_30 >> arr16[_l6]);
ubitcnt_30 = (ubitcnt_30 - arr16[_l6]);
}
}
}
torch::Tensor obj_arr56 = torch::empty({num_blocks,block_size}, at::kBool);
auto arr56 = obj_arr56.accessor<bool, 2>();
#pragma omp parallel for num_threads(16)
for (int _l8 = 0; _l8 < num_blocks; _l8 += 1) {
for (int _l9 = 0; _l9 < block_size; _l9 += 1) {
arr56[_l8][_l9] = (arr30[_l8][_l9] > 0);
}
}
torch::Tensor obj_arr63 = torch::empty({((input_size - num_blocks) - s21)}, at::kInt);
auto arr63 = obj_arr63.accessor<int, 1>();
for (int _l10 = 0; _l10 < ((input_size - num_blocks) - s21); _l10 += 1) {
arr63[_l10] = arr3[(_l10 + s21)];
}
torch::Tensor obj_arr67 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr67 = obj_arr67.accessor<int, 2>();
uint64_t ubits_67;
int ubitcnt_67;
int uinidx_67;
uint64_t umask_67;
ubits_67 = static_cast<uint64_t>(0);
ubitcnt_67 = 0;
uinidx_67 = 0;
for (int _l11 = 0; _l11 < num_blocks; _l11 += 1) {
for (int _l12 = 0; _l12 < block_size; _l12 += 1) {
if ((arr7[_l11] == 0)) {
arr67[_l11][_l12] = 0;
}
else {
if ((ubitcnt_67 < arr7[_l11])) {
ubits_67 = (ubits_67 | (static_cast<uint64_t>((arr63[uinidx_67] & 4294967295)) << ubitcnt_67));
ubitcnt_67 = (ubitcnt_67 + 32);
uinidx_67 = (uinidx_67 + 1);
}
else {
}
if ((ubitcnt_67 < arr7[_l11])) {
ubits_67 = (ubits_67 | (static_cast<uint64_t>((arr63[uinidx_67] & 4294967295)) << ubitcnt_67));
ubitcnt_67 = (ubitcnt_67 + 32);
uinidx_67 = (uinidx_67 + 1);
}
else {
}
if ((arr7[_l11] == 32)) {
umask_67 = static_cast<uint64_t>(4294967295);
}
else {
umask_67 = ((static_cast<uint64_t>(1) << arr7[_l11]) - static_cast<uint64_t>(1));
}
arr67[_l11][_l12] = static_cast<int>((ubits_67 & umask_67));
ubits_67 = (ubits_67 >> arr7[_l11]);
ubitcnt_67 = (ubitcnt_67 - arr7[_l11]);
}
}
}
torch::Tensor obj_arr93 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr93 = obj_arr93.accessor<int, 2>();
#pragma omp parallel for num_threads(16)
for (int _l13 = 0; _l13 < num_blocks; _l13 += 1) {
for (int _l14 = 0; _l14 < block_size; _l14 += 1) {
arr93[_l13][_l14] = (arr67[_l13][_l14] * -1);
}
}
torch::Tensor obj_arr100 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr100 = obj_arr100.accessor<int, 2>();
for (int _l15 = 0; _l15 < num_blocks; _l15 += 1) {
for (int _l16 = 0; _l16 < block_size; _l16 += 1) {
if (arr56[_l15][_l16]) {
arr100[_l15][_l16] = arr67[_l15][_l16];
}
else {
arr100[_l15][_l16] = arr93[_l15][_l16];
}
}
}
torch::Tensor obj_arr113 = torch::empty({num_blocks,block_size}, at::kInt);
auto arr113 = obj_arr113.accessor<int, 2>();
for (int _l17 = 0; _l17 < num_blocks; _l17 += 1) {
for (int _l18 = 0; _l18 < block_size; _l18 += 1) {
if ((_l18 == 0)) {
arr113[_l17][_l18] = arr100[_l17][_l18];
}
else {
arr113[_l17][_l18] = (arr113[_l17][(_l18 - 1)] + arr100[_l17][_l18]);
}
}
}
torch::Tensor obj_arr126 = torch::empty({num_blocks,block_size}, at::kFloat);
auto arr126 = obj_arr126.accessor<float, 2>();
for (int _l19 = 0; _l19 < num_blocks; _l19 += 1) {
for (int _l20 = 0; _l20 < block_size; _l20 += 1) {
arr126[_l19][_l20] = static_cast<float>(arr113[_l19][_l20]);
}
}
torch::Tensor obj_arr135 = torch::empty({num_blocks,block_size}, at::kFloat);
auto arr135 = obj_arr135.accessor<float, 2>();
#pragma omp parallel for num_threads(16)
for (int _l21 = 0; _l21 < num_blocks; _l21 += 1) {
for (int _l22 = 0; _l22 < block_size; _l22 += 1) {
arr135[_l21][_l22] = (arr126[_l21][_l22] * (eb * 2.0));
}
}
return obj_arr135;
}


class fzdemo_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", eb);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(fzdemo using generated torch compression/decompression code)");
    set(options, "pressio:abs", "absolute error bound passed to generated code");
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &eb);
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* output) override
  {
      try {
          if (real_input == nullptr || output == nullptr) {
              return set_error(1, "null input/output");
          }

          if (real_input->dtype() != pressio_float_dtype) {
              return set_error(1, "only float32 input is supported");
          }

          pressio_data input = domain_manager().make_readable(
              domain_plugins().build("malloc"),
              *real_input
          );

          if (input.dimensions().size() != 2) {
              return set_error(1, "this generated compressor expects 2D input");
          }

          const int num_blocks = static_cast<int>(input.dimensions()[1]);
          const int block_size = static_cast<int>(input.dimensions()[0]);

          auto begin = std::chrono::steady_clock::now();

          torch::Tensor input_tensor = pressio_to_tensor_f32(input);
          torch::Tensor compressed_tensor = concat_concat_ceil_log2_lufqwjkb(
              num_blocks,
              block_size,
              input_tensor,
              static_cast<float>(eb)
          );
          
          pressio_data packed = pack_compressed_tensor(compressed_tensor);
          *output = std::move(packed);

          auto end = std::chrono::steady_clock::now();
          compress_ms = std::chrono::duration<double, std::milli>(end - begin).count();
          return 0;
      } catch (std::exception const& ex) {
          return set_error(1, ex.what());
      }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
      try {
          if (input == nullptr || output == nullptr) {
              return set_error(1, "null input/output");
          }

          auto begin = std::chrono::steady_clock::now();

          auto unpacked = unpack_compressed_tensor(*input);

          const int num_blocks = static_cast<int>(202500);
          const int block_size = static_cast<int>(32);
          const int input_size = static_cast<int>(unpacked.payload.numel());

          torch::Tensor compressed_tensor = unpacked.payload;

          torch::Tensor restored_tensor = mul_cast_prefix_sum_maskiynoonyk(
              num_blocks,
              block_size,
              input_size,
              compressed_tensor,
              static_cast<float>(eb)
          );

          pressio_data restored = tensor_to_pressio_f32(restored_tensor);
          *output = std::move(restored);

          auto end = std::chrono::steady_clock::now();
          decompress_ms = std::chrono::duration<double, std::milli>(end - begin).count();
          return 0;
      } catch (std::exception const& ex) {
          return set_error(1, ex.what());
      }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "fzdemo"; }

  pressio_options get_metrics_results_impl() const override {
    return {
        {"fzdemo:compress_ms", compress_ms},
        {"fzdemo:decompress_ms", decompress_ms}
    };
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<fzdemo_compressor_plugin>(*this);
  }

  double compress_ms = 0;
  double decompress_ms = 0;
  double eb = 1e-4;
};

extern "C" BOOST_SYMBOL_EXPORT fzdemo_compressor_plugin plugin;
fzdemo_compressor_plugin plugin;

} } }
"""

compressor = lp.PressioCompressor.from_config(
    {
        "compressor_id": "pressio",
        "early_config": {
            "pressio:compressor": "poorjit",
            "poorjit:metric": "composite",
            "poorjit:generator": "template",
            "poorjit:pkgconfig": ["libpressio_cxx"],
            "poorjit:extra_args": [
                "-fopenmp",
                "-O3",
                "-I/data/backed_up/ssong10/Compiler-libpressio/spack/opt/spack/linux-cascadelake/boost-1.87.0-djs2gb3vjwxfpexn5qiop56r2t5kavlq/include",

                # torch include
                "-I/data/backed_up/ssong10/Compiler-libpressio/spack/opt/spack/linux-cascadelake/python-venv-1.0-us7klp5n5oxhcsra4vghgg4i24dqmozh/lib/python3.14/site-packages/torch/include",
                "-I/data/backed_up/ssong10/Compiler-libpressio/spack/opt/spack/linux-cascadelake/python-venv-1.0-us7klp5n5oxhcsra4vghgg4i24dqmozh/lib/python3.14/site-packages/torch/include/torch/csrc/api/include",

                # torch lib
                "-L/data/backed_up/ssong10/Compiler-libpressio/spack/opt/spack/linux-cascadelake/python-venv-1.0-us7klp5n5oxhcsra4vghgg4i24dqmozh/lib/python3.14/site-packages/torch/lib",
                "-Wl,-rpath,/data/backed_up/ssong10/Compiler-libpressio/spack/opt/spack/linux-cascadelake/python-venv-1.0-us7klp5n5oxhcsra4vghgg4i24dqmozh/lib/python3.14/site-packages/torch/lib",

                "-ltorch",
                "-ltorch_cpu",
                "-lc10",
            ],
            "composite:plugins": ["error_stat", "size", "time"],
            "template:source": SOURCE,
            "template:keys": [],
            "template:values": [],
        },
    }
)

options = compressor.get_options()
options["template:source"] = "...snip..."
pprint(options)

input = np.fromfile(
    Path(__file__).parent / "/data/not_backed_up/ssong10/compression_data/CESM-ATM/CLDHGH_1_1800_3600.dat",
    dtype=np.float32,
).reshape(202500, 32)

decompressed = np.empty_like(input)
compressed = compressor.encode(input)
decompressed = compressor.decode(compressed, decompressed)

pprint(compressor.get_metrics())
print("input shape:", input.shape)
print("compressed dtype:", compressed.dtype, "shape:", compressed.shape)
print("decompressed shape:", decompressed.shape)
print("max abs error:", np.max(np.abs(input - decompressed)))