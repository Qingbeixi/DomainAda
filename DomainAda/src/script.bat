@echo off

echo Starting process 1...
python src/reconstruction.py -source "s"
echo Process 1 complete.

echo Starting process 2...
python src/reconstruction.py -source "m"
echo Process 2 complete.

echo Starting process 3...
python src/reconstruction.py -source "l"
echo Process 3 complete.