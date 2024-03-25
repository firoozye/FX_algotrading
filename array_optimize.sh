#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=10:0:0
#$ -S /bin/bash
#$ -N abo_tuning
#$ -pe smp 5
#$ -R y
hostname
date
python3 ~/abo_research/abo/controllers/backtest_abo_strat.py --feat "macd" --obj "mse" --cross "GBPUSD"
echo "done"
