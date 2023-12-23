# cml-final-project-submission

This is the repository of our code submission for Cloud and ML.
We have collected here the scripts we have used and edited to enable memory profiling of LoRA and QLoRA on Stable Diffusion (and Llama), primarily leveraging huggingface diffusers and the LyCORIS codebases.

This repository features a collection of notebooks, configs, as well as some of the changes made to the libraries.


## SD Lycoris 

The script to train a Stable Diffusion model with LyCORIS can be found in `SD_lycoris.ipynb`. The script runs using accelerate through the train_network.py file. 
Various toml configurations can be changed in the `config/` folder.
While heavily modified, the runscript still depends on a number of the original author's functions which we have included in the `library/`. 
To run with different adaptors, simply pass the relevant config file as an argument to accelerate. eg. `--config_file configs/loha_config.toml`

## LLaMa LoRA & QLoRA

LLaMa script is accessible at `llama_finetuning_script.ipynb`, which should work out of the box without any local dependencies. The script is modified from huggingface/diffuser's example train scripts. Hyperparameters can be specified by overriding the values in the args namespace in the script. We have added some quick settings that can be set by passing method (one of `["lora8bit","qlora","full8bit"]`) to train(). 

