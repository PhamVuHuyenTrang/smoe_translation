# MoE-Multilingual

## Prepare envs

```
conda create -n fairseq python=3.8
conda activate fairseq

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

pip install fairscale==0.4.0 hydra-core==1.0.7 omegaconf==2.0.6
pip install boto3 zmq iopath nltk wandb pytorch_metric_learning
pip install sacrebleu[ja] sacrebleu[ko]
```

## Prepare data

1. Download original data
```
mkdir mmt_data && cd mmt_data
wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz && tar -xzf opus-100-corpus-v1.0.tar.gz
```

2. build sentencepiece
```
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
make install
ldconfig -v
```

3. preprocess data
refer to [nmt-multi](https://github.com/cordercorder/nmt-multi)
run preprocessing script `scripts/opus-100/data_process/multilingual_preprocess.sh`


## Training

Run `bash train_scripts/train_hrouting.sh 8192 8`


## Eval

Run `bash eval_scripts/eval.sh -d output/path_of_ckpt -n 8` 8 is the num of training gpus.
