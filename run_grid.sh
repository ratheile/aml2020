source $HOME/miniconda3/etc/profile.d/conda.sh
cd /cluster/home/inesb/aml2020  # Change inesb to your own ETH username

conda activate aml_env

bsub -R "rusage[mem=4000]" -W 24:00 -r "python main.py --user default --type grid --cfg project2/base_cfg.yml --slice project2/slice.yml --env env/cluster.yml"
