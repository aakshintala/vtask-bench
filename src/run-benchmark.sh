#! /bin/bash
#set -x #echo on

# Make a new directory for each run.
topdir=$(pwd)
outfile=$topdir"/vtask-bench-out-$(date +%Y-%m-%d-%H-%M-%S)"
touch $outfile

# the benchmark uses ints, so divide by sizeof(int)
# Tune this according to max memory on your GPU.
# GTX 970 I use has 4GiB so I set this to 1GiB
let totalDataMoved=1*1024*1024*1024;

# Min size is a page (4KiB)
let elemSize=4096;

# Max elemSize = 1GiB
while [[ $elemSize -lt 1073741824 ]]; do
	let numElems=$totalDataMoved/$elemSize;
	sudo timeout 6000 $topdir/vtask-bench -m -q $numElems -s $elemSize -t 1 | tee -a $outfile
	let oldcompute=0;
	for i in `seq 0 1 2`; do
		let inc=$((10**$i*2));
		let max=$((10*10**$i+1));
		let computeTime=$inc; # Time in us that simulates 'work'. Range: 10us to 1ms
		while [[ $computeTime -lt $max ]]; do # stop at 1 milliseconds
			if [[ $computeTime -ne $oldcompute ]]; then
			sudo timeout 6000 $topdir/vtask-bench -m -q $numElems -s $elemSize -t $computeTime | tee -a $outfile
			fi
			let oldcompute=$computeTime;
			let computeTime=$computeTime+$inc;
		done
	done
	let elemSize=$elemSize*4;
done
