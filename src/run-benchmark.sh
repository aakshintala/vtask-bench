#! /bin/bash
set -x #echo on

# Make a new directory for each run.
topdir=$(pwd)
outfile=$topdir"/vtask-bench-out-$(date +%Y-%m-%d-%H-%M-%S)"
touch outfile

# the benchmark uses ints, so divide by sizeof(int)
# Tune this according to max memory on your GPU.
# GTX 970 I use has 4GiB so I set this to 1GiB
let totalDataMoved=1*1024*1024*1024/4;

# Because each element is a 32-bit int, that's 4kb (1 page)
let elemSize=1024;

# Max elemSize = 1GiB
while [[ $elemSize -lt 1073741824 ]]; do
	let numElems=$totalDataMoved/$elemSize;
	let computeTime=1; # Time in microseconds that simulates 'work' Range 10us to 10ms
	while [[ $computeTime -lt 10000 ]]; do # stop at 1 milliseconds
		sudo timeout 6000 $topdir/vtask-bench -m -q $numElems -s $elemSize -t $computeTime | tee -a outfile
		let computeTime=$computeTime*10;
	done
	let elemSize=$elemSize*4;
done
