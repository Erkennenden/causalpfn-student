"""Display helpers for the code walkthrough."""
from pathlib import Path
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from pfn_interference.estimands import arm_mean_exposure

COLORS = ['#117c83', '#df8a39', '#5766ac', '#b34e65']
EFFECT_LABELS = ['Direct', 'Spillover', 'Total']


def theme():
    plt.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 150, 'figure.facecolor': '#ffffff',
        'axes.facecolor': '#f6f8fa', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#d5dee5',
        'axes.labelcolor': '#223344', 'text.color': '#223344',
        'xtick.color': '#526575', 'ytick.color': '#526575',
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titleweight': 'bold', 'axes.prop_cycle': plt.cycler(color=COLORS),
        'grid.color': '#dce5ea', 'grid.alpha': .7,
    })


def table(frame, precision=4):
    """Consistent, readable HTML table saved inside the notebook."""
    display(frame.style.format(precision=precision, na_rep='—').set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#eaf3f4'),
                                   ('color', '#174850'), ('padding', '9px 12px')]},
        {'selector': 'td', 'props': [('padding', '8px 12px'),
                                   ('border-bottom', '1px solid #e5ebef')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'),
                                      ('font-size', '14px')]},
    ]))


def cards(items):
    boxes = ''.join(
        '<div style="flex:1;min-width:145px;padding:18px;background:#edf6f6;'
        'border-radius:10px;border-top:3px solid #117c83">'
        f'<div style="font-size:12px;color:#527079">{html.escape(str(k))}</div>'
        f'<div style="font-size:23px;font-weight:700;color:#17324d;margin-top:5px">'
        f'{html.escape(str(v))}</div></div>' for k, v in items)
    display(HTML('<div style="display:flex;flex-wrap:wrap;gap:12px;margin:16px 0">'+boxes+'</div>'))


def show(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (name+'.png'), bbox_inches='tight')
    plt.show()
    plt.close(fig)


def conditional_exposures(degrees, probability=.5):
    degrees = np.asarray(degrees, dtype=int)
    lookup = {int(d): [arm_mean_exposure(int(d), probability, s) for s in (0, 1)]
              for d in np.unique(degrees)}
    return np.array([lookup[int(d)] for d in degrees])


def factual_frame(batch):
    """Only observable variables. No oracle, task-level coefficient or hidden parameters."""
    tokens = batch['tokens'][0].detach().cpu().numpy()
    adjacency = batch['adjacency'][0].detach().cpu().numpy()
    degrees = adjacency.sum(1).astype(int)
    x, treatment, outcome, exposure, degree_norm = tokens.T
    counts = np.rint(exposure * degrees).astype(int)
    return pd.DataFrame({'node': np.arange(len(x)), 'X': x, 'T': treatment.astype(int),
                         'Y': outcome, 'degree': degrees, 'K': counts, 'E': exposure,
                         'S': (counts > degrees//2).astype(int),
                         'neighbor_X': adjacency @ x / degrees,
                         'degree_norm': degree_norm})


def effects(mu):
    mu = np.asarray(mu)
    assert mu.ndim == 2 and mu.shape[1] == 4
    return np.column_stack([mu[:, 2]-mu[:, 0], mu[:, 3]-mu[:, 2], mu[:, 3]-mu[:, 0]])
