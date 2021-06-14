#!/bin/bash

declare -i DOP=`nproc`

START=`date +"%Y.%m.%d:%H.%M.%S"`

echo "used degree of parallelism: $DOP"

# some number of seconds between two polls of number of processes
declare -i SLEEP=30
#declare -i SLEEP=6
# current number of running processes
declare -i NOP=0
# count number of jobs
declare -i COUNT=0

#EXN=("quadrat.py" "Sobol_Verfahren.py")
EXN=("Sobol_Verfahren.py")

for program in ${EXN}
do
  for vector in {0..15}
  do
    echo "${program}"
    # prepare script to be submitted to pbs
    python ${program} ${vector} 0 0.25&
    #"${JOBDIR}"/./"${EXN}" "${BFILE}" 1 0 -3 "${Super_iter}" "${Iterator}" "${Reduce}" >> test/Test_"${Iterator}"_"${Reduce}"&

    PIDS="${PIDS} $1"
    ((COUNT = COUNT + 1))
    sleep 1
    NOP=`ps aux | grep ${EXN} | wc -l`
    echo "  nop = ${NOP}"
    while [[ $NOP -gt ${DOP} ]]
    do
      CURR=`date +"%Y.%m.%d:%H.%M.%S"`
      echo "sleep ${SLEEP} / ${START} / ${CURR} / (${NOP})"
      sleep ${SLEEP}
      NOP=`ps aux | grep ${EXN} | wc -l`
    done
  done
done

END=`date +"%Y.%m.%d:%H.%M.%S"`
echo "done submitting: ${END}"

wait ${PIDS}

END=`date +"%Y.%m.%d:%H.%M.%S"`
echo "done: start time: ${START}, end time: ${END}, no_jobs: ${COUNT}"
