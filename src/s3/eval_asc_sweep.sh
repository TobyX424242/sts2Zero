#!/bin/bash
# Evaluates one combat + macro pair at every ascension A0..A10 on the same fixed seeds (EVAL000000...), greedy macro,
# search 64 on every move. Three groups of ascensions run side by side, each group one ascension after another.
# usage: src/s3/eval_asc_sweep.sh COMBAT_WEIGHTS MACRO_FILE OUTDIR   (env: COUNT 150, N actors per group 7)
CW=$1; MW=$2; O=$3; COUNT=${COUNT:-150}; N=${N:-7}
W=$(cd "$(dirname "$0")/../.." && pwd); cd $W; mkdir -p $O
CONC=$(( (COUNT + N - 1) / N ))
group() { for a in "$@"; do
  COUNT=$COUNT ASC=$a CONC=$CONC CSERVERS=1 src/s3/eval_s3.sh $CW $MW $O/a$(printf %02d $a) $N "tf64:64" > $O/a$(printf %02d $a).log 2>&1
  echo "$(date '+%F %T') A$a done: $(grep -m1 '^== tf64' $O/a$(printf %02d $a)/summary.txt | cut -c1-120)" >> $O/progress.txt
done; }
echo "$(date '+%F %T') start: $CW $MW, $COUNT seeds per ascension, $N actors x 3 groups, conc $CONC" > $O/progress.txt
group 10 7 4 1 & group 9 6 3 0 & group 8 5 2 & wait
echo "$(date '+%F %T') all done" >> $O/progress.txt
