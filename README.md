# mag-diffusion
Bachelor project of Jeppe

## Dependencies

Install via
```python
conda create -n pmdd python=3.11 && conda activate pmdd
conda install magtense -c cmt-dtu-energy/label/cuda-12 -c nvidia/label/cuda-12.2.2
conda install -y notebook tqdm h5py
```

## Install local package

Navigate to main folder of repository `mag-diff` first, and then run the install command:

```bash
cd mag-diffusion
python -m pip install -e .
```