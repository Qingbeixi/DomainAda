@echo off

echo Starting process 1...
python src/regression_transfer.py -task "s to m" -ep 20 -lr 0.01
echo Process 1 complete.

echo Starting process 2...
python src/regression_transfer.py -task "s to l" -ep 20 -lr 0.01
echo Process 2 complete.

echo Starting process 3...
python src/regression_transfer.py -task "m to l" -ep 20 -lr 0.01
echo Process 3 complete.