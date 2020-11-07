source $HOME/miniconda3/etc/profile.d/conda.sh
cd $HOME/aml2020  # Change inesb to your own ETH username

conda activate aml2020
python main.py --env env/cluster.yml --cfg "$@"