CONFIG=$1
WEIGHT=$2
DST=$3
NAME=$4
GPU=$5
cp $WEIGHT $DST/$NAME.pth
cp $CONFIG $DST/$NAME.py
python tools/cal_prototypes/cal_prototype.py $CONFIG $WEIGHT $DST --gpu-id=$GPU