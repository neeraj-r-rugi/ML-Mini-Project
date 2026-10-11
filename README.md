

##  How to Run the Project

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/neeraj-r-rugi/ML-Mini-Project.git
cd ML-Mini-Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Model Artifacts
Open and run all cells in [`model_training.ipynb`](model_training.ipynb) to train the models and export the serialized artifacts to `./Models/`:
```bash
jupyter lab model_training.ipynb
```

### 3. Launch the Streamlit App
```bash
streamlit run demo.py
```
The application will launch in your browser at `http://localhost:8501`.


