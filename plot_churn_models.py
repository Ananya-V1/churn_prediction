import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Chart colors
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
BAR = '#2a78d6'
BLUES = LinearSegmentedColormap.from_list('blues', ['#f3f8fe', '#9ec5f4', '#3987e5', '#1c5cab', '#104281'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': TEXT_PRIMARY, 'axes.labelcolor': TEXT_SECONDARY,
    'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
    'font.size': 11,
})

# Models to plot: file prefix -> (display name, saved model)
MODELS = {
    'gb': ('Gradient boosting', 'gb_churn_model.pkl'),
    'rf': ('Random forest', 'rf_churn_model.pkl'),
}

# Load the encoders
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Rebuild the same test set as the training scripts
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data['Gender'] = label_encoder_gender.transform(data['Gender'])
geo_encoded = onehot_encoder_geo.transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

X = data.drop('Exited', axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs('plots', exist_ok=True)
labels = ['Stayed', 'Churned']


def plot_confusion_matrix(model, name, prefix):
    probs = model.predict_proba(X_test)[:, 1]

    # Confusion matrices at the default threshold and at 0.3
    thresholds = [0.5, 0.3]
    matrices = [confusion_matrix(y_test, probs > t) for t in thresholds]
    vmax = max(m.max() for m in matrices)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, t, cm in zip(axes, thresholds, matrices):
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                row_pct = cm[i, j] / cm[i].sum()
                ink = SURFACE if cm[i, j] > vmax * 0.45 else TEXT_PRIMARY
                ax.text(j, i - 0.08, f'{cm[i, j]:,}', ha='center', va='center', fontsize=18, color=ink, fontweight='bold')
                ax.text(j, i + 0.18, f'{row_pct:.0%} of actual {labels[i].lower()}', ha='center', va='center', fontsize=9.5, color=ink)
        recall = cm[1, 1] / cm[1].sum()
        precision = cm[1, 1] / cm[:, 1].sum()
        ax.set_title(f'Threshold {t}  ·  recall {recall:.0%}  ·  precision {precision:.0%}', fontsize=11.5, color=TEXT_PRIMARY, pad=10)
        ax.set_xticks([0, 1], [f'Predicted\n{l.lower()}' for l in labels])
        ax.set_yticks([0, 1], [f'Actually\n{l.lower()}' for l in labels])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 2px surface gap between cells
        ax.axhline(0.5, color=SURFACE, linewidth=2)
        ax.axvline(0.5, color=SURFACE, linewidth=2)

    fig.suptitle(f'{name} churn model: confusion matrix (test set, 2,000 customers)', x=0.02, ha='left', fontsize=13.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f'plots/{prefix}_confusion_matrix.png', dpi=150)
    plt.close(fig)


def plot_feature_importance(model, name, prefix):
    importance = pd.Series(model.feature_importances_, X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(importance.index, importance.values, color=BAR, height=0.62, edgecolor=SURFACE, linewidth=1)
    for bar, value in zip(bars, importance.values):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f'{value:.3f}', va='center', fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_xlabel('Importance (share of total, sums to 1)')
    ax.set_xlim(0, importance.max() * 1.15)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ['top', 'right', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.set_title(f'{name} churn model: feature importance', loc='left', fontsize=13.5, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(f'plots/{prefix}_feature_importance.png', dpi=150)
    plt.close(fig)


for prefix, (name, path) in MODELS.items():
    with open(path, 'rb') as file:
        model = pickle.load(file)
    plot_confusion_matrix(model, name, prefix)
    plot_feature_importance(model, name, prefix)
    print(f'Saved plots/{prefix}_confusion_matrix.png and plots/{prefix}_feature_importance.png')
