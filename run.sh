if [ $# -lt 3 ]; then
	echo usage: $0 DATASET MODEL SEED [arguments]
	exit
fi

DATASET=$1
MODEL=$2
SEED=$3
shift 3

python src/main.py \
	--conf-file settings/${MODEL}-${DATASET}-tiara.json \
	--seed $SEED \
	$@
