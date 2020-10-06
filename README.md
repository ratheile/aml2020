# aml2020
Advanced Machine Learning 2020

# How to run things:


## Bash \ ZSH
First make sure you have a default configuration in 'env/env.yml'. This file is ignored via .gitignore and should be configured locally to your likings. Copy the template to start.

You may also give an environment via `--env`.

Run a single file:
```
 python main.py --user raffi --cfg project1_raffi/base_cfg.yml
```


Use yml_gen to generate several config files / or create them manually:
```
 python yml_gen.py --hparams project1_raffi/slice.yml --cfg project1_raffi/base_cfg.yml
 python main.py --user raffi --dir experiments/test1
```

## VSCode
If you use vscode you can use my debug configurations:

```{json}
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Main",
      "type": "python",
      "request": "launch",
      "program": "main.py",
      "console": "internalConsole",
      "args": [
        "--user", "raffi",
        "--cfg", "project1_raffi/base_cfg.yml"
      ]
    },
    {
      "name": "Python: Yaml Generator",
      "type": "python",
      "request": "launch",
      "program": "yml_gen.py",
      "console": "internalConsole",
      "args": [
        "--hparams", "project1_raffi/slice.yml",
        "--cfg", "project1_raffi/base_cfg.yml"
      ]
    }
  ]
}
```
