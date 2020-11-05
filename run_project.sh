source $HOME/miniconda3/etc/profile.d/conda.sh
cd /cluster/home/inesb/aml2020  # Change inesb to your own ETH username

# conda env create --file environment.yml
# conda activate aml_env

bsub -J "CreateYmlFiles" python yml_gen.py --cfg project2/base_cfg.yml --hparams project2/slice.yml 

# Why you shouldn't parse ls output: https://mywiki.wooledge.org/ParsingLs
for file in experiments/test1/project2*; do # will need changing if we change the path or the naming of the yml files.
    bsub -w "done(CreateYmlFiles)" -n 4 -R "rusage[mem=10000]" "python main.py --cfg $file" # To understand the dollar sign + @: https://stackoverflow.com/questions/5163144/what-are-the-special-dollar-sign-shell-variables
done
