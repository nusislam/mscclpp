// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/concurrency_device.hpp>
#include <string>

#include "common.hpp"

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

static void* inputBuff = nullptr;
static void* resultBuff = nullptr;
static void* scratchBuff = nullptr;
static void* scratchPacketBuff = nullptr;
static void* putPacketBuff = nullptr;
static void* getPacketBuff = nullptr;

namespace {
auto isUsingHostOffload = [](int kernelNum) { return kernelNum == 3; };
constexpr uint64_t MAGIC = 0xdeadbeef;
}  // namespace

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::SimpleProxyChannel> constProxyChans[32];

__constant__ DeviceHandle<mscclpp::ProxyChannel> constRawProxyChan[16];

__constant__ DeviceHandle<mscclpp::SmChannel> constSmChans[512];
__constant__ DeviceHandle<mscclpp::SmChannel> constSmOutOfPlaceChans[16];

__device__ uint64_t globalFlag;

std::unordered_map<DeviceHandle<mscclpp::SimpleProxyChannel>*, int > channelInfos;

__global__ void allgather0(int rank, size_t nelemsPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::SimpleProxyChannel> proxyChan = constProxyChans[warpId];

  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if (threadIdx.x % WARP_SIZE == 0) {
    proxyChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  }
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if (threadIdx.x % WARP_SIZE == 0) proxyChan.flush();

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if (threadIdx.x % WARP_SIZE == 0) proxyChan.wait();
}

__device__ void localAllGather(DeviceHandle<mscclpp::SimpleProxyChannel> proxyChan, int rank, int nRanksPerNode,
                               int remoteRank, uint64_t offset, uint64_t size, bool flushAfterSignal = true) {
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nRanksPerNode
  // and waits for data from GPU rank (i-1) % nRanksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nRanksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  for (int i = 1; i < nRanksPerNode; i++) {
    if ((remoteRank % nRanksPerNode) == ((rank + i) % nRanksPerNode)) {
      // put your data to GPU (rank+i) % nRanksPerNode and signal in one call
      if (flushAfterSignal && (threadIdx.x % WARP_SIZE) == 0) proxyChan.putWithSignalAndFlush(offset, size);
      if (!flushAfterSignal && (threadIdx.x % WARP_SIZE) == 0) proxyChan.putWithSignal(offset, size);
    }
    // wait for the data from GPU (rank-i) % nRanksPerNode to arrive
    if ((remoteRank % nRanksPerNode) == ((rank - i + nRanksPerNode) % nRanksPerNode)) {
      if ((threadIdx.x % WARP_SIZE) == 0) proxyChan.wait();
    }
#if defined(__HIP_PLATFORM_AMD__)
    // NOTE: we actually need a group barrier here for better performance, but __syncthreads() is still correct.
    __syncthreads();
#else
    asm volatile("bar.sync %0, %1;" ::"r"(11), "r"((nRanksPerNode - 1) * WARP_SIZE) : "memory");
#endif
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherSm(int rank, int nRanksPerNode, int startRankChunkIndex, uint64_t offsetInRankChunk,
                                 uint64_t rankChunkSize, uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const size_t nPeer = nRanksPerNode;

  const size_t peerIdx = blockIdx.x % nPeer;
  
  

  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;
  if (threadIdx.x == 0 && peerLocalBlockIdx == 0) {
    constSmChans[peerIdx].signal();
    constSmChans[peerIdx].wait();
  }
  deviceSyncer.sync(nBlocks);
  if (peerIdx == rank) return;

  size_t offset = (peerLocalBlockIdx*nRanksPerNode + (rank % nRanksPerNode)) * rankChunkSize;
  constSmChans[peerIdx].put<16, false>(offset, rankChunkSize, threadIdx.x, blockDim.x);
}

__global__ void allgather1(int rank, int nRanksPerNode, size_t nelemsPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::SimpleProxyChannel> proxyChan = constProxyChans[warpId];

  localAllGather(proxyChan, rank, nRanksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                 nelemsPerGPU * sizeof(int));
}

__global__ void allgather2(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::SimpleProxyChannel> proxyChan = constProxyChans[warpId];

  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = 3;

  // Step 1
  // local allgather
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(proxyChan, rank, nRanksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                   nelemsPerGPU * sizeof(int), false);
  }
  // cross-node exchange
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    // opposite side
    if ((threadIdx.x % WARP_SIZE) == 0)
      proxyChan.putWithSignal(rank * nelemsPerGPU * sizeof(int),
                              (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
    if ((threadIdx.x % WARP_SIZE) == 0) proxyChan.wait();
  }

  // sync here to make sure IB flush dose not block the CUDA IPC traffic
  __syncthreads();
  // need to flush ib channel here to avoid cq overflow. since we won't change send suffer after send, we don't need
  // to flush for IPC channel.
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    if ((threadIdx.x % WARP_SIZE) == 0) proxyChan.flush();
  }
  __syncthreads();

  // Step 2
  // local allgather
  int otherNghr = (rank + nRanksPerNode) % worldSize;
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(proxyChan, rank, nRanksPerNode, remoteRank, otherNghr * nelemsPerGPU * sizeof(int),
                   (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int), false);
  }

  // cross-node exchange
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    // opposite side
    if ((threadIdx.x % WARP_SIZE) == 0)
      proxyChan.putWithSignal((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                              nelemsPerGPU / pipelineSize * sizeof(int));
    if ((threadIdx.x % WARP_SIZE) == 0) proxyChan.wait();
  }

  __syncthreads();
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    if ((threadIdx.x % WARP_SIZE) == 0) proxyChan.flush();
  }
  __syncthreads();

  // Step 3
  // local allgather
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(proxyChan, rank, nRanksPerNode, remoteRank,
                   (otherNghr * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                   nelemsPerGPU / pipelineSize * sizeof(int));
  }
}

__global__ void allgather3() {
  int warpId = threadIdx.x / WARP_SIZE;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::ProxyChannel> proxyChan = constRawProxyChan[warpId];

  int tid = threadIdx.x;
  __syncthreads();
  if (tid == 0) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = MAGIC;
    trigger.snd = 0;
    // offload all the work to the proxy
    uint64_t currentFifoHead = proxyChan.fifo_.push(trigger);
    // wait for the work to be done in cpu side
    proxyChan.fifo_.sync(currentFifoHead);
  }
  if (tid % WARP_SIZE == 0) {
    proxyChan.wait();
  }
}

__global__ void allgather4(void *buff, void* putPktBuf, void* getPktBuf, int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data
  int num_nodes = worldSize / nRanksPerNode;
  const size_t nBlocksForLocalAllGather = gridDim.x;
  int peer_src, peer_dst;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  uint32_t flag = (uint32_t)globalFlag;

  //const size_t nBlocksForLocalAllGather = gridDim.x;
  const size_t rankChunkSize = nelemsPerGPU * sizeof(int);
  const int startRankIndexInLocalNode = (rank / nRanksPerNode) * nRanksPerNode;
  const int startRankIndexInPeerNode = (peerRank / nRanksPerNode) * nRanksPerNode;


  size_t step1Bytes = nelemsPerGPU * sizeof(int);
  int cur_count = nelemsPerGPU;
  int pof2 = 1;
  int offset = 0;

  int2* src = (int2*)((char*)buff + rank * nelemsPerGPU * sizeof(int));
  
  if (num_nodes > 1) {
	/*if (threadIdx.x == 0 && blockIdx.x == 0) 
		printf("Entering %d\n", rank);*/
   while (pof2 <= num_nodes / 2) {
        peer_src = (rank + pof2*nRanksPerNode) % worldSize;
        peer_dst = (rank - pof2*nRanksPerNode + worldSize) % worldSize;
        DeviceHandle<mscclpp::SimpleProxyChannel>& proxyChan_src = constProxyChans[peer_src % nRanksPerNode];
        DeviceHandle<mscclpp::SimpleProxyChannel>& proxyChan_dst = constProxyChans[peer_dst % nRanksPerNode];

	//DeviceHandle<mscclpp::SimpleProxyChannel>& proxyChan = nullptr;	
	size_t nPkts = cur_count / 2;  // 2 elems per packet, assume nelems is even
  	size_t pktBytes = nPkts * sizeof(mscclpp::LLPacket);
	size_t pktBufOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
	//size_t pktBufOffset = 0;

  	mscclpp::LLPacket* getPktPtr = (mscclpp::LLPacket*)((char*)getPktBuf + pktBufOffset);
  	mscclpp::LLPacket* putPktPtr = (mscclpp::LLPacket*)((char*)putPktBuf + pktBufOffset);

  	int2* res = (int2*)((char*)buff + rank * nelemsPerGPU * sizeof(int) + cur_count*sizeof(int));

	for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
		putPktPtr[idx].write(src[idx].x, src[idx].y, flag);
	}
   	
	constexpr int nBlocksPhase2 = 1;

  	deviceSyncer.sync(nBlocksForLocalAllGather);
   	//if (blockIdx.x >= nBlocksPhase2) return;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
                //proxyChan_dst.putWithSignal(peer_dst*nelemsPerGPU*sizeof(int) + cur_count*sizeof(int), rank*nelemsPerGPU*sizeof(int), cur_count*sizeof(int));
		proxyChan_dst.put(pktBufOffset, pktBytes);
		if ((flag & 63) == 0)
                proxyChan_dst.flush();
        }

	//if (blockIdx.x < nBlocksPhase2) {
	    //for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * nBlocksPhase2) {
	    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {

      	   	uint2 data1 = getPktPtr[idx].read(flag);

      	   	res[idx].x = (int)data1.x;
      	   	res[idx].y = (int)data1.y;
            }
	//}

	//if (threadIdx.x == 0 && blockIdx.x == 0) {
	    flag = flag + 1;
  	//}
	cur_count *= 2;
        pof2 *= 2;
	deviceSyncer.sync(nBlocksForLocalAllGather);	
     }

	int shift_by = rank / nRanksPerNode;

	if (blockIdx.x > 0 && blockIdx.x <= (num_nodes - 1)) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;			
		if (shift_by == 0) {
			int target_offset = blockIdx.x * nRanksPerNode + rank;
			int src_offset = (rank + blockIdx.x) * nelemsPerGPU * sizeof(int);
				
			constSmChans[rank].put<16, false>(target_offset*nelemsPerGPU*sizeof(int), src_offset, nelemsPerGPU*sizeof(int), threadIdx.x, blockDim.x);
		} else {
			int target_offset = (blockIdx.x * nRanksPerNode + rank);

			if (target_offset >= worldSize)
				target_offset = (blockIdx.x * nRanksPerNode + rank) % worldSize;

			int src_offset = (rank + blockIdx.x) * nelemsPerGPU * sizeof(int);

			constSmChans[rank].put<16, false>(target_offset*nelemsPerGPU*sizeof(int), src_offset, nelemsPerGPU*sizeof(int), threadIdx.x, blockDim.x);			   
		}

    	}
        //deviceSyncer.sync(nBlocksForLocalAllGather);
	localAllGatherSm(rank, nRanksPerNode, 0, 0, rankChunkSize, rankChunkSize,
                   nBlocksForLocalAllGather);
		
  } else {
  	localAllGatherSm(rank, nRanksPerNode, 0, 0, rankChunkSize, rankChunkSize,
                   nBlocksForLocalAllGather);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag = flag;
  }

}

__global__ void __launch_bounds__(1024, 1) allgather5(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = constSmChans + chanOffset;

  if (wid < nPeer && lid == 0) {
    smChans[wid].relaxedSignal();
    smChans[wid].wait();
  }
  __syncthreads();
  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread;
  if (bytes >= nThread * 64) {
    unitBytesPerThread = 64;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (wid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].get<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (gWid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].get<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      smChans[peerIdx].get<16, true>(offset, remainBytes, lid, WARP_SIZE);
    }
  }
}

__global__ void __launch_bounds__(1024, 1) allgather6(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = constSmChans + chanOffset;

  if (wid < nPeer && lid == 0) {
    smChans[wid].relaxedSignal();
    smChans[wid].wait();
  }
  __syncthreads();
  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread;
  if (bytes >= nThread * 64) {
    unitBytesPerThread = 64;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    // const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * rank + (wid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].put<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    // const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].put<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * rank + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      smChans[peerIdx].put<16, true>(offset, remainBytes, lid, WARP_SIZE);
    }
  }
}

__global__ void __launch_bounds__(1024, 1) allgather7(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  // const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = constSmOutOfPlaceChans;

  const uint32_t flag = (uint32_t)globalFlag;
  // if (wid < nPeer && lid == 0) {
  //   smChans[wid].relaxedSignal();
  //   smChans[wid].wait();
  // }
  // __syncthreads();
  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread = 8;
  // if (bytes >= nThread * 64) {
  //   unitBytesPerThread = 64;
  // } else {
  //   unitBytesPerThread = 16;
  // }
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  // double buffering
  const size_t scratchOffset = (flag & 1) ? 0 : bytesPerGPU * nRanksPerNode * 2;

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    // const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * rank + (wid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }


  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (wid / nPeer) * unitBytesPerWarp;
    // if (lid == 0) printf("get %p, rank %zu, wid %zu, remoteRankLocalIndex %zu, offset %zu\n", smChans[peerIdx].getPacketBuffer_, rank, wid, remoteRankLocalIndex, offset);
    smChans[peerIdx].getPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    // const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (gWid / nPeer) * unitBytesPerWarp;
    smChans[peerIdx].getPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }

  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * rank + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      smChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, remainBytes, lid, WARP_SIZE, flag);
    }
  }
  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      smChans[peerIdx].getPackets(scratchOffset + offset * 2, offset, remainBytes, lid, WARP_SIZE, flag);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

__global__ void __launch_bounds__(1024, 1) allgather8(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x / 2;
  const bool isPut = blockIdx.x < nBlock;

  const size_t tid = threadIdx.x + (blockIdx.x % nBlock) * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  // const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = constSmOutOfPlaceChans;

  const uint32_t flag = (uint32_t)globalFlag;
  // if (wid < nPeer && lid == 0) {
  //   smChans[wid].relaxedSignal();
  //   smChans[wid].wait();
  // }
  // __syncthreads();
  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread = 8;
  // if (bytes >= nThread * 64) {
  //   unitBytesPerThread = 64;
  // } else {
  //   unitBytesPerThread = 16;
  // }
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  // double buffering
  const size_t scratchOffset = (flag & 1) ? 0 : bytesPerGPU * nRanksPerNode * 2;

  if (isPut) {
    if (nLoop > 0) {
      // First loop unrolling
      const size_t peerIdx = wid % nPeer;
      // const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      const size_t offset = bytesPerGPU * rank + (wid / nPeer) * unitBytesPerWarp;
      // smChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
      mscclpp::putPackets(smChans[peerIdx].dst_, scratchOffset + offset * 2, smChans[peerIdx].src_, offset,
                          unitBytesPerWarp, lid, WARP_SIZE, flag);
    }


    for (size_t i = 1; i < nLoop; ++i) {
      const size_t gWid = wid + i * nWarp;
      const size_t peerIdx = gWid % nPeer;
      // const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
      // smChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
      mscclpp::putPackets(smChans[peerIdx].dst_, scratchOffset + offset * 2, smChans[peerIdx].src_, offset,
                          unitBytesPerWarp, lid, WARP_SIZE, flag);
    }

    if (bytes % unitBytes > 0) {
      const size_t gWid = wid + nLoop * nWarp;
      const size_t peerIdx = gWid % nPeer;
      const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
      const size_t offset = bytesPerGPU * rank + offsetWithinRank;
      const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                    ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                    : unitBytesPerWarp;
      if (remainBytes > 0) {
        // smChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, remainBytes, lid, WARP_SIZE, flag);
        mscclpp::putPackets(smChans[peerIdx].dst_, scratchOffset + offset * 2, smChans[peerIdx].src_, offset,
                            remainBytes, lid, WARP_SIZE, flag);
      }
    }
  } else {
    if (nLoop > 0) {
      // First loop unrolling
      const size_t peerIdx = wid % nPeer;
      const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      const size_t offset = bytesPerGPU * remoteRankLocalIndex + (wid / nPeer) * unitBytesPerWarp;
      // if (lid == 0) printf("get %p, rank %zu, wid %zu, remoteRankLocalIndex %zu, offset %zu\n", smChans[peerIdx].getPacketBuffer_, rank, wid, remoteRankLocalIndex, offset);
      // smChans[peerIdx].getPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
      mscclpp::getPackets(smChans[peerIdx].getPacketBuffer_, scratchOffset + offset * 2, smChans[peerIdx].src_, offset,
                          unitBytesPerWarp, lid, WARP_SIZE, flag);
    }

    for (size_t i = 1; i < nLoop; ++i) {
      const size_t gWid = wid + i * nWarp;
      const size_t peerIdx = gWid % nPeer;
      const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      const size_t offset = bytesPerGPU * remoteRankLocalIndex + (gWid / nPeer) * unitBytesPerWarp;
      // smChans[peerIdx].getPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
      mscclpp::getPackets(smChans[peerIdx].getPacketBuffer_, scratchOffset + offset * 2, smChans[peerIdx].src_, offset,
                          unitBytesPerWarp, lid, WARP_SIZE, flag);
    }

    if (bytes % unitBytes > 0) {
      const size_t gWid = wid + nLoop * nWarp;
      const size_t peerIdx = gWid % nPeer;
      const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
      const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
      const size_t offset = bytesPerGPU * remoteRankLocalIndex + offsetWithinRank;
      const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                    ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                    : unitBytesPerWarp;
      if (remainBytes > 0) {
        // smChans[peerIdx].getPackets(scratchOffset + offset * 2, offset, remainBytes, lid, WARP_SIZE, flag);
        mscclpp::getPackets(smChans[peerIdx].getPacketBuffer_, scratchOffset + offset * 2, smChans[peerIdx].src_, offset,
                            remainBytes, lid, WARP_SIZE, flag);
      }
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

class AllGatherProxyService : public mscclpp::BaseProxyService {
 public:
  AllGatherProxyService(int worldSize, int rank, int cudaDevice);
  void startProxy() override { proxy_->start(); }
  void stopProxy() override { proxy_->stop(); }
  void setSendBytes(size_t sendBytes) { this->sendBytes_ = sendBytes; }
  void addRemoteMemory(mscclpp::RegisteredMemory memory) { remoteMemories_.push_back(memory); }
  void setLocalMemory(mscclpp::RegisteredMemory memory) { localMemory_ = memory; }
  mscclpp::SemaphoreId buildAndAddSemaphore(mscclpp::Communicator& communicator,
                                            std::shared_ptr<mscclpp::Connection> connection) {
    semaphores_.push_back(std::make_shared<mscclpp::Host2DeviceSemaphore>(communicator, connection));
    return semaphores_.size() - 1;
  }
  std::vector<DeviceHandle<mscclpp::ProxyChannel>> proxyChannels() {
    std::vector<DeviceHandle<mscclpp::ProxyChannel>> result;
    for (auto& semaphore : semaphores_) {
      result.push_back(mscclpp::deviceHandle(mscclpp::ProxyChannel(0, semaphore, proxy_)));
    }
    return result;
  }

 private:
  int worldSize_;
  int rank_;
  int cudaDevice_;
  size_t sendBytes_;

  std::shared_ptr<mscclpp::Proxy> proxy_;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores_;
  std::vector<mscclpp::RegisteredMemory> remoteMemories_;
  mscclpp::RegisteredMemory localMemory_;

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger triggerRaw);
};

AllGatherProxyService::AllGatherProxyService(int worldSize, int rank, int cudaDevice)
    : worldSize_(worldSize),
      rank_(rank),
      cudaDevice_(cudaDevice),
      sendBytes_(0),
      proxy_(
          std::make_shared<mscclpp::Proxy>([&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); },
                                           [&]() {
                                             int deviceNumaNode = getDeviceNumaNode(cudaDevice_);
                                             numaBind(deviceNumaNode);
                                           })) {}

mscclpp::ProxyHandlerResult AllGatherProxyService::handleTrigger(mscclpp::ProxyTrigger triggerRaw) {
  size_t offset = rank_ * sendBytes_;
  if (triggerRaw.fst != MAGIC) {
    // this is not a valid trigger
    throw std::runtime_error("Invalid trigger");
  }
  for (int i = 0; i < worldSize_; i++) {
    int r = (rank_ + i) % worldSize_;
    if (r == rank_) {
      continue;
    }
    int index = (r < rank_) ? r : r - 1;
    semaphores_[index]->connection()->write(remoteMemories_[index], offset, localMemory_, offset, sendBytes_);
    semaphores_[index]->signal();
  }
  bool flushIpc = false;
  for (auto& semaphore : semaphores_) {
    auto conn = semaphore->connection();
    if (conn->transport() == mscclpp::Transport::CudaIpc && !flushIpc) {
      // since all the cudaIpc channels are using the same cuda stream, we only need to flush one of them
      conn->flush();
      flushIpc = true;
    }
    if (mscclpp::AllIBTransports.has(conn->transport())) {
      conn->flush();
    }
  }
  return mscclpp::ProxyHandlerResult::FlushFifoTailAndContinue;
}

class AllGatherTestColl : public BaseTestColl {
 public:
  AllGatherTestColl() = default;
  ~AllGatherTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff, std::vector<void*> tmpBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void AllGatherTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int nRanksPerNode = args.nRanksPerNode;
  const int kernelNum = args.kernelNum;
  int num_nodes = worldSize / nRanksPerNode;

  int nBlocks;
  int nThreads;
  if (kernelNum == 4) {
    //nBlocks = 21;
	  //printf("num_nodes = %d, nRanksPerNode = %d\n", num_nodes, nRanksPerNode);
    nBlocks = num_nodes * nRanksPerNode;
    //nBlocks = nRanksPerNode - 1;

    nThreads = 1024;
  } else if (kernelNum == 5) {
    nBlocks = 24;
    nThreads = 1024;
  } else if (kernelNum == 6) {
    nBlocks = 24;
    nThreads = 1024;
  } else if (kernelNum == 7) {
    nBlocks = 4;
    nThreads = 896;
  } else if (kernelNum == 8) {
    nBlocks = 2;
    nThreads = 896;
  } else {
    nBlocks = 1;
    nThreads = WARP_SIZE * (worldSize - 1);
  }
  if (kernelNum == 0) {
    allgather0<<<nBlocks, nThreads, 0, stream>>>(rank, paramCount_);
  } else if (kernelNum == 1) {
    allgather1<<<nBlocks, nThreads, 0, stream>>>(rank, nRanksPerNode, paramCount_);
  } else if (kernelNum == 2) {
    allgather2<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 3) {
    allgather3<<<nBlocks, nThreads, 0, stream>>>();
  } else if (kernelNum == 4) {
    allgather4<<<nBlocks, nThreads, 0, stream>>>((int*)inputBuff, putPacketBuff, getPacketBuff, rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 5) {
    allgather5<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 6) {
    allgather6<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 7) {
    allgather7<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 8) {
    allgather8<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  }
}

void AllGatherTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff, std::vector<void*> tmpBuff) {
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  if (tmpBuff.size() != 1) std::runtime_error("unexpected error");

  int rank = args.rank;
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), 0);
  for (size_t i = 0; i < recvCount_; i++) {
    int val = i + 1;
    if (i / sendCount_ == (size_t)rank) {
      dataHost[i] = val;
    } else {
      dataHost[i] = 0;
    }
  }
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), recvCount_ * typeSize_, cudaMemcpyHostToDevice));
  CUDATHROW(cudaMemcpy(tmpBuff[0], dataHost.data(), recvCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = static_cast<int>(i) + 1;
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void AllGatherTestColl::getBw(const double deltaSec, double& algBw, double& busBw) {
  double baseBw = (double)(paramCount_ * typeSize_ * worldSize_) / 1.0E9 / deltaSec;

  algBw = baseBw;
  double factor = ((double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllGatherTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  size_t base = (count / worldSize_);
  sendCount_ = base;
  recvCount_ = base * worldSize_;
  paramCount_ = base;
  expectedCount_ = recvCount_;
  if (isUsingHostOffload(kernelNum_)) {
    auto service = std::dynamic_pointer_cast<AllGatherProxyService>(chanService_);
    service->setSendBytes(sendCount_ * typeSize_);
  }
  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

std::vector<KernelRestriction> AllGatherTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes, alignedBytes}
          {0, "allgather0", true, 1, 4 * worldSize_},
          {1, "allgather1", false, 1, 4 * worldSize_},
          {2, "allgather2", true, 3, 4 * worldSize_},
          {3, "allgather3", true, 1, 4 * worldSize_},
          {4, "allgather4", true, 1, 4 * worldSize_ /*use ulong2 to transfer data*/},
          {5, "allgather5", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {6, "allgather6", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {7, "allgather7", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {8, "allgather8", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/}};
}

class AllGatherTestEngine : public BaseTestEngine {
 public:
  AllGatherTestEngine(const TestArgs& args);
  ~AllGatherTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

  std::vector<void*> getSendBuff() override;
  std::vector<void*> getTmpBuff() override;

  void* getRecvBuff() override;
  void* getScratchBuff() override;
  std::shared_ptr<mscclpp::BaseProxyService> createProxyService() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int> tmpBuff_;
  
  std::shared_ptr<mscclpp::LLPacket> putPacketBuff_;
  std::shared_ptr<mscclpp::LLPacket> getPacketBuff_;


  std::shared_ptr<int[]> expectedBuff_;
  std::shared_ptr<mscclpp::LLPacket> scratchPacketBuff_;
  std::vector<mscclpp::SmChannel> smChannels_;
  std::vector<mscclpp::SmChannel> smOutOfPlaceChannels_;
};

AllGatherTestEngine::AllGatherTestEngine(const TestArgs& args) : BaseTestEngine(args, "allgather") {}

void AllGatherTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::allocExtSharedCuda<int>(2*args_.maxBytes / sizeof(int));
  tmpBuff_ = mscclpp::allocExtSharedCuda<int>(args_.maxBytes / sizeof(int));

  inputBuff = sendBuff_.get();
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);

  if (args_.kernelNum == 4) {
	const size_t nPacket = (2*args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
	const size_t packetBuffNelem = nPacket * 2;
    	const size_t scratchBuffNelem = nPacket * 2; /*original data & reduced result *2 double buffering*/;

	printf("Allocate Buf nelem = %zu\n", packetBuffNelem);
    	putPacketBuff_ = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(packetBuffNelem);
    	getPacketBuff_ = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(packetBuffNelem);
    	scratchPacketBuff_ = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(scratchBuffNelem);

    	putPacketBuff = putPacketBuff_.get();
    	getPacketBuff = getPacketBuff_.get();
  }
  if (args_.kernelNum == 7 || args_.kernelNum == 8) {
    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    // 2x for double-buffering, scratchBuff used to store original data and reduced results
    const size_t scratchBuffNelem = nPacket * 2 /*original data & reduced result */ * 2 /* double buffering*/;
    scratchPacketBuff_ = mscclpp::allocExtSharedCuda<mscclpp::LLPacket>(scratchBuffNelem);
  }
}

void AllGatherTestEngine::setupConnections() {
  std::vector<DeviceHandle<mscclpp::SimpleProxyChannel>> devProxyChannels;
  if (!isUsingHostOffload(args_.kernelNum)) {
    if (args_.kernelNum == 4) {
	 const size_t nPacket = (2*args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

	 const size_t scratchPacketBuffBytes = nPacket * 2 * sizeof(mscclpp::LLPacket);
	 const size_t packetBuffBytes = nPacket * 2 * sizeof(mscclpp::LLPacket);
	 printf("In mesh conn for kernel 4 packetBuffBytes = %zu\n", packetBuffBytes);

	 // build proxyChannels
	 setupMeshConnections(smChannels_, devProxyChannels, sendBuff_.get(), args_.maxBytes, putPacketBuff_.get()       	, packetBuffBytes, getPacketBuff_.get(), packetBuffBytes, nullptr, 0);

	if (devProxyChannels.size() > sizeof(constProxyChans) / sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>)) {
	      std::runtime_error("unexpected error");
    	}
    	CUDATHROW(cudaMemcpyToSymbol(constProxyChans, devProxyChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>) * devProxyChannels.size()));

	// build smChannels
	setupMeshConnections(smChannels_, sendBuff_.get(), args_.maxBytes, nullptr, 0, ChannelSemantic::PUT, 64);

	std::vector<DeviceHandle<mscclpp::SmChannel>> smChannelHandles(smChannels_.size());
	if (smChannels_.size() > sizeof(constSmChans) / sizeof(DeviceHandle<mscclpp::SmChannel>)) {
      		std::runtime_error("unexpected error");
    	}
    	std::transform(smChannels_.begin(), smChannels_.end(), smChannelHandles.begin(),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    	CUDATHROW(cudaMemcpyToSymbol(constSmChans, smChannelHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::SmChannel>) * smChannelHandles.size()));
    } else {
    setupMeshConnections(devProxyChannels, sendBuff_.get(), args_.maxBytes);

    if (devProxyChannels.size() > sizeof(constProxyChans) / sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>)) {
      std::runtime_error("unexpected error");
    }
    CUDATHROW(cudaMemcpyToSymbol(constProxyChans, devProxyChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::SimpleProxyChannel>) * devProxyChannels.size()));

    setupMeshConnections(smChannels_, sendBuff_.get(), args_.maxBytes, nullptr, 0, ChannelSemantic::PUT, 64);

    //setupMeshConnections(smChannels_, tmpBuff_.get(), args_.maxBytes, nullptr, 0, ChannelSemantic::PUT, 64);

    std::vector<DeviceHandle<mscclpp::SmChannel>> smChannelHandles(smChannels_.size());
    if (smChannels_.size() > sizeof(constSmChans) / sizeof(DeviceHandle<mscclpp::SmChannel>)) {
      std::runtime_error("unexpected error");
    }
    std::transform(smChannels_.begin(), smChannels_.end(), smChannelHandles.begin(),
                   [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
    CUDATHROW(cudaMemcpyToSymbol(constSmChans, smChannelHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::SmChannel>) * smChannelHandles.size()));
    }
    if (args_.kernelNum == 7 || args_.kernelNum == 8) {
      const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      const size_t scratchPacketBuffBytes = nPacket * 2 * 2 * sizeof(mscclpp::LLPacket);
      setupMeshConnections(smOutOfPlaceChannels_, sendBuff_.get(), args_.maxBytes, scratchPacketBuff_.get(),
                           scratchPacketBuffBytes);
      std::vector<DeviceHandle<mscclpp::SmChannel>> smOutOfPlaceChannelHandles(smOutOfPlaceChannels_.size());
      if (smOutOfPlaceChannels_.size() > sizeof(constSmOutOfPlaceChans) / sizeof(DeviceHandle<mscclpp::SmChannel>)) {
        std::runtime_error("unexpected error");
      }
      std::transform(smOutOfPlaceChannels_.begin(), smOutOfPlaceChannels_.end(), smOutOfPlaceChannelHandles.begin(),
                     [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
      CUDATHROW(cudaMemcpyToSymbol(constSmOutOfPlaceChans, smOutOfPlaceChannelHandles.data(),
                                   sizeof(DeviceHandle<mscclpp::SmChannel>) * smOutOfPlaceChannelHandles.size()));
    }
  } else {
    printf("host offload\n");

    auto service = std::dynamic_pointer_cast<AllGatherProxyService>(chanService_);
    setupMeshConnections(devProxyChannels, sendBuff_.get(), args_.maxBytes, nullptr, 0,
                         [&](std::vector<std::shared_ptr<mscclpp::Connection>> conns,
                             std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>& remoteMemories,
                             const mscclpp::RegisteredMemory& localMemory) {
                           std::vector<mscclpp::SemaphoreId> semaphoreIds;
                           for (size_t i = 0; i < conns.size(); ++i) {
                             service->buildAndAddSemaphore(*comm_, conns[i]);
                             service->addRemoteMemory(remoteMemories[i].get());
                           }
                           service->setLocalMemory(localMemory);
                           comm_->setup();
                         });
    auto proxyChannels = service->proxyChannels();
    if (proxyChannels.size() > sizeof(constRawProxyChan) / sizeof(DeviceHandle<mscclpp::ProxyChannel>)) {
      std::runtime_error("unexpected error");
    }
    CUDATHROW(cudaMemcpyToSymbol(constRawProxyChan, proxyChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::ProxyChannel>) * proxyChannels.size()));
  }
}

std::shared_ptr<mscclpp::BaseProxyService> AllGatherTestEngine::createProxyService() {
  if (isUsingHostOffload(args_.kernelNum)) {
    return std::make_shared<AllGatherProxyService>(args_.totalRanks, args_.rank, args_.gpuNum);
  } else {
    return std::make_shared<mscclpp::ProxyService>();
  }
}

std::vector<void*> AllGatherTestEngine::getSendBuff() { return {sendBuff_.get()}; }
std::vector<void*> AllGatherTestEngine::getTmpBuff() { return {tmpBuff_.get()}; }

void* AllGatherTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllGatherTestEngine::getRecvBuff() {
  // in-place operation reuse the send buffer
  return sendBuff_.get();
}

void* AllGatherTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllGatherTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllGatherTestColl>(); }
