#! /bin/bash
set -x #echo on

# Make a new directory for each run.
topdir=$(pwd)
benchmarkDir=$topdir"/benchmark-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir $benchmarkDir
cd $benchmarkDir

# the benchmark uses ints, so divide by sizeof(int)
# Tune this according to max memory on your GPU.
# GTX 970 I use has 4GiB so I set this to 2GiB
let totalDataMoved=2*1024*1024*1024/4;

# Because each element is a 32-bit int, that's 4kb (1 page)
let elemSize=1024;

# Max elemSize = 1GiB
while [[ $elemSize -lt 1073741824 ]]; do
	let numElems=$totalDataMoved/$elemSize;
	let computeTime=1; # Time in microseconds that simulates 'work' Range 10us to 10ms
	while [[ $computeTime -lt 10000 ]]; do # stop at 10 milliseconds
		subDir=$numElems-$elemSize-$computeTime
		mkdir $subDir
		cd $subDir
		sudo timeout 6000 $topdir/p2pbenchmark -q $numElems -s $elemSize -t $computeTime | tee timing
		cd ..
		let computeTime=$computeTime*10;
	done
	let elemSize=$elemSize*4;
done
