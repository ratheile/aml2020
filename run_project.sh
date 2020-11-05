source $HOME/miniconda3/etc/profile.d/conda.sh
cd /cluster/home/inesb/aml2020  # Change inesb to your own ETH username

conda activate aml_env
python main.py --env env/cluster.yml --cfg "$@"