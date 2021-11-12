# Install the requirements and login to wandb.
pip install --upgrade -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
# wandb login [your wandb api-key]

# Install NVIDIA apex library
git clone https://github.com/NVIDIA/apex
sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
rm -rf apex
