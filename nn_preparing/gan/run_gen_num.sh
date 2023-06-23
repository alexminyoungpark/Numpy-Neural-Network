#!/bin/bash

set echo on

python3.9 ./gen_num.py --batch-size 600 --epoch-size 10
python3.9 ./gen_num.py --batch-size 3000 --epoch-size 50
python3.9 ./gen_num.py --batch-size 6000 --epoch-size 100
python3.9 ./gen_num.py --batch-size 30000 --epoch-size 500
python3.9 ./gen_num.py --batch-size 60000 --epoch-size 1000
