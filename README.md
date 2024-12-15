# Intro
Adaptive Equalizer via self-supervised learning.

# How to Run
### `fit` stage 
```bash
python src/main.py fit -c configs/config.yaml -n debug-fit-run -v debug-version
```
If using `wandb` for logging, change `"project"` key in `cli_module/rich_wandb.py`
If you want to access log directory in your `LightningModule`, you can access as follows.
```python
log_root_dir = self.logger.log_dir or self.logger.save_dir
```


### Model Checkpoint
One can save model checkpoints using `Lightning Callbacks`. 
It contains model weight, and other state_dict for resuming train.  
There are several ways to save ckpt files at either local or cloud.

1. Just leave everything in default, ckpt files will be saved locally. (at `logs/${name}/${version}/fit/checkpoints`)

2. If you want to save ckpt files as `wandb` Artifacts, add the following config. (The ckpt files will be saved locally too.)
```yaml
trainer:
  logger:
    init_args:
      log_model: all
```
3. If you want to save ckpt files in cloud rather than local, you can change the save path by adding the config. (The ckpt files will **NOT** be saved locally.)
```yaml
model_ckpt:
  dirpath: gs://bucket_name/path/for/checkpoints
```

#### `AsyncCheckpointIO` Plugins
You can set async checkpoint saving by providing config as follows.  
```yaml
trainer:
  plugins:
    - AsyncCheckpointIO
```



#### Automatic Batch Size Finder
Just add `BatchSizeFinder` callbacks in the config
```yaml
trainer:
  callbacks:
    - class_path: BatchSizeFinder
```
Or add them in the cmdline.
```bash
python src/main.py fit -c configs/config.yaml --trainer.callbacks+=BatchSizeFinder
```

##### NEW! `tune.py` for lr_find and batch size find
```bash
python src/tune.py -c configs/config.yaml
```
NOTE: No subcommand in cmdline

#### Resume
Basically all logs are stored in `logs/${name}/${version}/${job_type}` where `${name}` and `${version}` are configured in yaml file or cmdline. 
`{job_type}` can be one of `fit`, `test`, `validate`, etc.
  

### `test` stage
```bash
python src/main.py test -c configs/config.yaml -n debug-test-run -v debug-version --ckpt_path YOUR_CKPT_PATH
```
