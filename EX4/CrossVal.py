import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm



pio.renderers.default='browser'

np.random.seed(0)
X_ = np.random.randn(500, 10000)
y = np.random.randint(2, size=500)
scores = []
k_list = list(range(1, 10001, 1000))

# Create a pipeline that scales the data then trains a support vector classifier
classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
for i, k in enumerate(k_list):
    print('%d from %d of list itmes are checked'%(i+1, len(k_list)))
    X_selected = SelectKBest(k=k).fit_transform(X_, y)

    #scores.append(cross_val_score(classifier_pipeline, X_selected, y, cv=10).mean())
    scores = cross_val_score(classifier_pipeline, X_selected, y, cv=10)

fig = go.Figure()

fig.add_trace(go.Scatter(
        x=k_list,
        y=scores,
        line=dict(
                    color='#0173b2',
                    width=3
                ),
        name='Accuracy 1'
        ))

fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=18),
                 xaxis= {'title': 'Number of features'},
                yaxis = {'title': 'Accuracy'},
                yaxis_range=[0,1])
fig.show()
