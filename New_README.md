In your local machine, we recommend to first create a virtual environment:
```bash
conda create -n wrag python=3.10
conda activate wrag
git clone https://github.com/jmnian/WRAG.git
cd WRAG
conda install -c conda-forge openjdk=22
pip install -r requirements.txt
```