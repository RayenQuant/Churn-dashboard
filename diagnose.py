import traceback
import sys

print(f"Python: {sys.version}")
print()

steps = [
    ("numpy", "import numpy; print('numpy', numpy.__version__)"),
    ("pandas", "import pandas; print('pandas', pandas.__version__)"),
    ("sklearn", "import sklearn; print('sklearn', sklearn.__version__)"),
    ("xgboost", "import xgboost; print('xgboost', xgboost.__version__)"),
    ("shap", "import shap; print('shap', shap.__version__)"),
    ("plotly", "import plotly; print('plotly', plotly.__version__)"),
    ("streamlit", "import streamlit; print('streamlit', streamlit.__version__)"),
    ("faker", "import faker; print('faker OK')"),
    ("pipeline import", "import pipeline; print('pipeline imported OK')"),
    ("pipeline ensure_db", "import pipeline; pipeline.ensure_db(); print('DB created OK')"),
    ("utils import", "import utils; print('utils imported OK')"),
]

for name, code in steps:
    print(f"Testing: {name} ...", end=" ")
    try:
        exec(code)
    except Exception:
        print(f"FAILED")
        traceback.print_exc()
        print()
        print("=== STOPPING HERE — this is your error ===")
        break
