@echo off

echo Starting process 1...
python src/regression.py -task "s" -ep 30 -lr 0.01
python src/regression.py -task "s" -ep 40 -lr 0.01
python src/regression.py -task "s" -ep 50 -lr 0.01
python src/regression.py -task "s" -ep 30 -lr 0.05
python src/regression.py -task "s" -ep 40 -lr 0.05
python src/regression.py -task "s" -ep 50 -lr 0.05
echo Process 1 complete.

echo Starting process 2...
python src/regression.py -task "m" -ep 30 -lr 0.01
python src/regression.py -task "m" -ep 40 -lr 0.01
python src/regression.py -task "m" -ep 50 -lr 0.01
python src/regression.py -task "m" -ep 30 -lr 0.05
python src/regression.py -task "m" -ep 40 -lr 0.05
python src/regression.py -task "m" -ep 50 -lr 0.05
echo Process 2 complete.

echo Starting process 3...
python src/regression.py -task "l" -ep 30 -lr 0.01
python src/regression.py -task "l" -ep 40 -lr 0.01
python src/regression.py -task "l" -ep 50 -lr 0.01
python src/regression.py -task "l" -ep 30 -lr 0.05
python src/regression.py -task "l" -ep 40 -lr 0.05
python src/regression.py -task "l" -ep 50 -lr 0.05
echo Process 3 complete.