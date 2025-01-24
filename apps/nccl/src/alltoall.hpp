// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLGATHER_HPP_
#define ALLGATHER_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    alltoall6(void* sendbuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, size_t channelOutOffset,
               size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * gridDim.x;
  const size_t nWarp = nThread / WARP_SIZE;
  
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = smChannels + chanOffset;

  /*const size_t peerIdx = blockIdx.x % nPeer;
  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;*/

  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;

  deviceSyncer.sync(gridDim.x);

  

  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
  }
}

template <bool IsOutOfPlace, typename T>
cudaError_t alltoall(T* buff, [[maybe_unused]] T* scratch, [[maybe_unused]] T* resultBuff,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, size_t channelOutOffset, int rank,
                      int nRanksPerNode, int worldSize, size_t nelems, cudaStream_t stream) {
  int nBlocks = 28;
  if (nelems <= 4096) {
    nBlocks = 7;
  } else if (nelems <= 32768) {
    nBlocks = 14;
  } else if (nelems >= 2097152) {
    nBlocks = 35;
  }
  alltoall6<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>((void*)buff, smChannels, channelOutOffset, rank, worldSize,
                                                         nRanksPerNode, nelems * sizeof(T) / sizeof(int));
  return cudaGetLastError();
}

#endif  // ALLGATHER_HPP_
