# vtask-bench

Time spent transferring data (Td) to a GPU, and the time spent computing (Tc) on that data on the
GPU have an intricate relationship: If Td greatly exceeds Tc, it maybe worth considering pipelining
data offload and computation, or even if offloading to the GPU is profitable at all in the first
place. If Tc greatly exceeds Td, data movement is typically not a first-order concern. However, in
the middle where Td is close to Tc, there is an interesting case to explore, especially in a
virtualiazed environment that involves copying data to and from the guest
(e.g., [AvA](https://github.com/utcs-scea/ava/)).

This is a synthetic benchmark that attempts to explore the space of workloads whose Td and Tc are
within a magnitude of each other. At this time, I can't find any real world applications or
non-synthetic benchmarks that exhibit such behavior, hence the motivation for a synthetic benchmark.

This benchmark primarily copies some tunable amount of data to a GPU, runs a kernel that primarily
loops until a tunable amount of time has expired, and then copies the data back to the host. In a
native setting, this workload does nothing useful.

However, consider an application running in a virtualized setting (like AvA) where the user-space
API is interposed for virtualization. If Td were in the same ballpark as Tc, any increases in Td
(due to extra copying in the virtualization layer, for example) would have an impact on the
end-to-end performance of the application. This benchmark provides a synthetic workload that can be
used to quantify the impact of the extra copying of data in such a case.

Things to fix if you want to use this benchmark:
Makefile: make sure the generated cubin has the right arch version for the GPU you have.
