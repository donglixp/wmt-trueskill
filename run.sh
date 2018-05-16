METRIC=$1

cd src
for i in {0..100}
do
    cat ../data/turk.csv | python infer_TS.py ../result/$METRIC$i -d 0 -s 5 -m $METRIC -min 0 -max 2000000 &
done
wait

cd ../eval
python cluster.py fr-en ../result/$METRIC -n 100 -by-rank -pdf
