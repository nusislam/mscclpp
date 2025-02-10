# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from mscclpp.language import *
from mscclpp.language.collectives import AllToAll
from mscclpp.language.buffer import Buffer
from mscclpp.language.types import ChannelType, ReplicationPolicy


def alltoall_multinodes_allpair(gpus, gpus_per_node, instances):
    """
    Implements a multi-node allgather collective using an allpairs algorithm with MSCCL++ DSL.
    @param gpus: Total number of GPUs
    @param gpus_per_node: Number of GPUs per node
    Steps:
    1. Each rank sends a chunk to all other ranks' scratch buffers using packet format.
    2. Copy the chunk from the scratch buffer to the output buffer using packet format.
    """
    collective = AllToAll(gpus, 1, False)
    with MSCCLPPProgram(
        "alltoall_multinodes_allpair",
        collective,
        gpus,
        instances,
        protocol="Simple",
        replication_policy=ReplicationPolicy.interleaved,
        num_threads_per_block=1024,
    ):

        for n in range(gpus):
            for peer in range(gpus):
                c = chunk(n, Buffer.input, peer, 1)
                if n == peer:
                    c.copy(n, Buffer.output, n, sendtb=peer)
                elif n // gpus_per_node == peer // gpus_per_node:
                    c.put(peer, Buffer.output, n, sendtb=peer, chan_type=ChannelType.sm)
                else:
                    c.put(peer, Buffer.output, n, sendtb=peer, chan_type=ChannelType.port)
        # explicit barrier
            r = rank(n)
            r.barrier(tb_list=list(range(gpus)))
            for peer in range(gpus):
                if n == peer:
                    continue
                elif n // gpus_per_node == peer // gpus_per_node: 
                    c.signal(peer, Buffer.output, n, sendtb=peer, chan_type=ChannelType.sm)
                else:
                    c.signal(peer, Buffer.output, n, sendtb=peer, chan_type=ChannelType.port)

        for n in range(gpus):
            for peer in range(gpus):
                c = chunk(n, Buffer.output, peer, 1)
                if n == peer:
                    continue
                elif n // gpus_per_node == peer // gpus_per_node: 
                    c.wait(peer, Buffer.input, peer, recvtb=peer, chan_type=ChannelType.sm)
                else:
                    c.wait(peer, Buffer.input, peer, recvtb=peer, chan_type=ChannelType.port)
        Json()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument("num_gpus", type=int, help="number of gpus")
parser.add_argument("gpus_per_node", type=int, help="number of gpus")
parser.add_argument("instances", type=int, help="number of instances")

args = parser.parse_args()

alltoall_multinodes_allpair(
    args.num_gpus,
    args.gpus_per_node,
    args.instances,
)
