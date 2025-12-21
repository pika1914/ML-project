# ML-project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.imput import simpelImputer
data_set=pd.read_csv("/content/aisles.csv")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
