<h1 align="center"> LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model </h1> 

<h5 align="center"><em>Dilxat Muhtar*, Zhenshi Li* , Feng Gu, Xueliang Zhang, and Pengfeng Xiao</em>
</br>(*Equal Contribution)</h5>


<figure>
<div align="center">
<img src=https://pumpkintypora.oss-cn-shanghai.aliyuncs.com/lhrsbot.png width="20%">
</figure>
<p align="center">
  <a href="#news">News</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#Preparation">Preparation</a> |
  <a href="#Demo">Demo</a> | 
  <a href="#acknowledgement">Acknowledgement</a> |
  <a href="#statement">Statement</a>
</p >

## News
+ **\[Sep 21 2024\]:** We are excited to release **LHRS-Bot-Nova**, a powerful new model with an upgraded base model, enriched training data, and an improved training recipe. Please check out the nova branch [here](https://github.com/NJU-LHRS/LHRS-Bot/tree/nova) and follow the instructions to start using the model. The model checkpoint is available on [huggingface](https://huggingface.co/LHRS/LHRS-Bot-Nova/tree/main).
+ **\[Jul 15 2024\]:** We updated our paper at [arxiv](https://arxiv.org/abs/2402.02544).
+ **\[Jul 12 2024\]:** We post the missing part in our paper (some observations, considerations, and lessons) in [Zhihu](https://zhuanlan.zhihu.com/p/708415355) (In Chinese, please contact us if you need English version).
+ **\[Jul 09 2024\]:** We have released our evaluation benchmark [LHRS-Bench](https://huggingface.co/datasets/PumpkinCat/LHRS_Data/tree/main/LHRS-Bench).
+ **\[Jul 02 2024\]:** Our paper has been accepted by ECCV 2024! We have open-sourced our training script and training data. Please follow the training instruction belown and [data preparation](./DataPrepare/README.md).
+ **\[Feb 21 2024\]:** We have updated our evaluation code. Any advice are welcom!
+ **\[Feb 7 2024\]:** Model weights are now available on both Google Drive and Baidu Disk.
+ **\[Feb 6 2024\]:** Our paper now is available at [arxiv](https://arxiv.org/abs/2402.02544).
+ **\[Feb 2 2024\]:** We are excited to announce the release of our code and model checkpoint! Our dataset and training recipe will be update soon!

## Introduction

We are excited to introduce **LHRS-Bot-Nova**, an enhanced version of LHRS-Bot with LLaMA-3 as base model, MoE bridge layer design, and improved caption and strong vision undetstanding ability. 

<figure>
<div align="center">
<img src=assets/architecture.png width="100%">
</div>
</figure>


## Preparation

### Installation

1. Clone this repository.

    ~~~shell
    git clone git@github.com:NJU-LHRS/LHRS-Bot.git
    cd LHRS-Bot
	git checkout nova
    ~~~

2. Create a new virtual enviroment

    ~~~shell
    conda create -n lhrs python=3.10
    conda activate lhrs
    ~~~

3. Install dependences and our package

    ~~~shell
    pip install -e .
    ~~~

### Checkpoints

+ LLaMA3-8B-Instruct

    + Automaticaly download:

        Our framework is designed to automatically download the checkpoint when you initiate training or run a demo. However, there are a few preparatory steps you need to complete:

        1. Request the LLaMA3-8B-Instruct models from [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

        2. After your request been processed, login to huggingface using your personal access tokens:

            ~~~shell
            huggingface-cli login
            (Then paste your access token and press Enter)
            ~~~

        3. Done!

    + Mannually download:

        + Download all the files from [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

        + Change the following line of each file to your downloaded directory:

            + `/Config/multi_modal_stage{1, 2, 3}.yaml`

                ~~~yaml
                ...
                text:
                	...
                  path: ""  # TODO: Direct to your directory
                ...
                ~~~

            + `/Config/multi_modal_eval.yaml`

                ~~~yaml
                ...
                text:
                	...
                  path: ""  # TODO: Direct to your directory
                ...
                ~~~

+ LHRS-Bot Checkpoints:

	+ Download from our [huggingface repo](https://huggingface.co/LHRS/LHRS-Bot-Nova/tree/main)
	+ ⚠️ Ensure that the `TextLoRA` folder is located in the same directory as `FINAL.pt`. The name `TextLoRA` should remain unchanged. Our framework will automatically detect the version perceiver checkpoint and, if possible, load and merge the LoRA module.

## Demo

+ Online Web UI demo with gradio:

    ~~~shell
    python lhrs_webui.py \
         -c Config/multi_modal_eval.yaml \           # config file
         --checkpoint-path ${PathToCheckpoint}.pt \  # path to checkpoint end with .pt
         --server-port 8000 \                        # change if you need
         --server-name 127.0.0.1 \                   # change if you need
         --share                                     # if you want to share with other
    ~~~

+ Command line demo:

  ~~~shell
  python cli_qa.py \
       -c Config/multi_modal_eval.yaml \                 # config file
       --model-path ${PathToCheckpoint}.pt \             # path to checkpoint end with .pt
       --image-file ${TheImagePathYouWantToChat} \       # path to image file (Only Single Image File is supported)
       --accelerator "gpu" \                             # change if you need ["mps", "cpu", "gpu"]
       --temperature 0.85 \
       --max-new-tokens 512
  ~~~

+ Inference:

    + Classification

        ~~~shell
        python main_cls.py \
             -c Config/multi_modal_eval.yaml \                 # config file
             --model-path ${PathToCheckpoint}.pt \             # path to checkpoint end with .pt
             --data-path ${ImageFolder} \                      # path to classification image folder
             --accelerator "gpu" \                             # change if you need ["mps", "cpu", "gpu"]
             --workers 4 \
             --enabl-amp True \
             --output ${YourOutputDir}                         # Path to output (result, metric etc.)
             --batch-size 8 \
        ~~~

    + Visual Grounding

        ~~~shell
        python main_vg.py \
             -c Config/multi_modal_eval.yaml \                 # config file
             --model-path ${PathToCheckpoint}.pt \             # path to checkpoint end with .pt
             --data-path ${ImageFolder} \                      # path to image folder
             --accelerator "gpu" \                             # change if you need ["mps", "cpu", "gpu"]
             --workers 2 \
             --enabl-amp True \
             --output ${YourOutputDir}                         # Path to output (result, metric etc.)
             --batch-size 1 \                                  # It's better to use batchsize 1, since we find batch inference
             --data-target ${ParsedLabelJsonPath}              # is not stable.
        ~~~

    + Visual Question Answering

        ~~~shell
        python main_vqa.py \
             -c Config/multi_modal_eval.yaml \                 # config file
             --model-path ${PathToCheckpoint}.pt \             # path to checkpoint end with .pt
             --data-path ${Image} \                            # path to image folder
             --accelerator "gpu" \                             # change if you need ["mps", "cpu", "gpu"]
             --workers 2 \
             --enabl-amp True \
             --output ${YourOutputDir}                         # Path to output (result, metric etc.)
             --batch-size 1 \                                  # It's better to use batchsize 1, since we find batch inference
             --data-target ${ParsedLabelJsonPath}              # is not stable.
             --data-type "HR"                                  # choose from ["HR", "LR"]
        ~~~

        

## Acknowledgement

+ We gratitude to the following repositories for their wonderful works:
    + [LLaVA](https://github.com/haotian-liu/LLaVA)
    + [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
    + [LLaMA](https://github.com/facebookresearch/llama)


## Statement

+ If you find our work is useful, please give us 🌟 in GitHub and consider cite our paper:

    ~~~tex
    @misc{2402.02544,
    Author = {Dilxat Muhtar and Zhenshi Li and Feng Gu and Xueliang Zhang and Pengfeng Xiao},
    Title = {LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model},
    Year = {2024},
    Eprint = {arXiv:2402.02544},
    }
    ~~~

+ Licence: Apache
