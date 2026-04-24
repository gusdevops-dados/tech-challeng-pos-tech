# Notebooks

Sugestão para o seu notebook atual:

- Renomeie para `01_eda_nps.ipynb`
- Mantenha nele as análises conceituais, gráficos exploratórios e interpretações.
- Aos poucos, substitua blocos repetidos por imports do `src`.

Exemplo:

```python
from src.data.make_dataset import load_data
from src.features.build_features import add_analysis_features, prepare_model_data

df = load_data("../data/raw/desafio_nps_fase_1.csv")
df = add_analysis_features(df)
X, y, features, encoder = prepare_model_data(df)
```
