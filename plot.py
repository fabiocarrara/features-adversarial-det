import argparse
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

sns.set_style('darkgrid')


def evaluate(y_hat, y):
    fpr, tpr, thr = metrics.roc_curve(y, y_hat)
    auc = metrics.auc(fpr, tpr)
    
    # EER accuracy
    fnr = 1 - tpr
    eer_thr = thr[np.nanargmin(np.absolute(fnr - fpr))]
    eer_accuracy = metrics.accuracy_score(y, y_hat > eer_thr)
    eer = (eer_accuracy, eer_thr)
    
    # Best TPR-FPR
    dist = fpr**2 + (1-tpr)**2
    best = np.argmin(dist)
    best = fpr[best], tpr[best], thr[best], auc

    return eer_accuracy, eer_thr, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Adversarial Detections')
    # DATA PARAMS
    parser.add_argument('data', help='Data to plot')
    args = parser.parse_args()
    
    data = np.load(args.data)
    y_hat, y, keep = data['y_hat'], data['y'], data['keep']

    data = {
        'kept': data['keep'],
        'type': np.array(('fgsm', 'auth', 'noise', 'step_fgsm', 'iter_fgsm')).repeat(1000)
    }

    data = pd.DataFrame(data)
    data['modified'] = ~data['type'].isin(('auth', 'noise'))

    data['group'] = data['type']

    good = ~data['modified'] & data['kept']
    data.loc[good, 'group'] = 'auth'

    errors = ~data['modified'] & ~data['kept']
    data.loc[errors, 'group'] = 'errors'

    failed = data['modified'] & ~data['kept']
    data.loc[failed, 'group'] = 'failed'

    # print(data.groupby(['type', 'group']).count())
    # print(data[data['kept']].groupby(['type', 'group']).count())

    kept = data.loc[data['kept'], ['type', 'modified', 'group']]
    kept['pred'] = y_hat

    eer_accuracy, eer_thr, auc = evaluate(kept['pred'], kept['modified'])
    print('Global EER Accuracy: {:3.2%}'.format(eer_accuracy))
    kept['eer_pred'] = kept['pred'] > eer_thr

    accs = kept.groupby('group').apply(lambda x: '{:3.2%}'.format(metrics.accuracy_score(x['eer_pred'], x['modified'])))
    print(accs)
    
    colors = np.array(('r', 'b'))
    
    n_samples = len(y)
    plt.scatter(np.arange(n_samples), y_hat, color=colors[y], marker='.')
    plt.savefig('plot.pdf')
