import argparse
import os

import matplotlib

matplotlib.use('Agg')
import glob2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

palette = sns.color_palette("Paired")
sns.set('paper', 'whitegrid', palette, font='serif', font_scale=1, rc={
    'text.usetex': True,
    'legend.frameon': True,
})


def evaluate(y_hat, y):
    fpr, tpr, thr = metrics.roc_curve(y, y_hat)
    auc = metrics.auc(fpr, tpr)

    # EER accuracy
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer_thr = thr[eer_idx]
    eer_accuracy = metrics.accuracy_score(y, y_hat > eer_thr)
    eer = (eer_accuracy, eer_thr)

    # Best TPR-FPR
    dist = fpr ** 2 + (1 - tpr) ** 2
    best = np.argmin(dist)
    best = fpr[best], tpr[best], thr[best], auc

    return fpr, tpr, thr, eer_idx, eer_accuracy, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Adversarial Detectors ROC')
    # DATA PARAMS
    parser.add_argument('runs', help='Folder containing runs')
    parser.add_argument('-a', '--adversarial-info', default='adversarial.csv', help='CSV with adversarial info')
    parser.add_argument('-o', '--output', help='Output plot file')

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.runs, 'rocs.pdf')

    adv = pd.read_csv(args.adversarial_info)

    runs = glob2.glob(os.path.join(args.runs, '**', 'predictions.csv'))

    plot_fn = plt.semilogx
    # plot_fn = plt.plot
    
    plt.figure(figsize=(5,5))

    aucs = []

    def sort_key_fn(x):
        p = os.path.join(os.path.dirname(x), 'params.csv')
        p = pd.read_csv(p)
        keys = p[['centroids', 'mlp', 'distance']]
        keys['centroids'] = keys['centroids'].map(lambda x: 1 if 'centroid' in x else 0)
        return keys.values[0].tolist()

    runs = sorted(runs, key=sort_key_fn)

    broken_down_aucs = pd.DataFrame()

    for run in runs:
        data = pd.read_csv(run, index_col=0)
        params = os.path.join(os.path.dirname(run), 'params.csv')
        params = pd.read_csv(params)
        if params.hidden.values[0] != 100 or params.bidir.values[0]: continue

        fpr, tpr, thr, eer_idx, eer_accuracy, auc = evaluate(data.pred, data.target)
        # print(params)
        # print('Global EER Accuracy: {:3.2%}'.format(eer_accuracy))
        # print('EER-TPR: {:3.2%}'.format(tpr[eer_idx]))

        #label = '{}, $\mathbf{{P}}={}$, $d={}$ (AUC={:4.3f})'.format(
        label = '{} + {} + ${}$ (AUC={:4.3f})'.format(
            ('MLP' if params.mlp.values[0] else 'LSTM'),
            ('M' if 'medoid' in params.centroids.values[0] else 'C'),
            ('\cos' if params.distance.values[0] == 'cosine' else 'L_2'),
            auc)

        aucs.append(auc)
        linestyle = '--' if 'centroid' in params.centroids.values[0] else '-'
        plot_fn(fpr, tpr, linestyle, label=label,)
        # plt.scatter(fpr[eer_idx], tpr[eer_idx])

        data = data.join(adv.set_index('Path'), on='image')
        data.Attack = data.Attack.fillna('auth')

        auths = data[data['Attack'] == 'auth']
        data.Eps = data.Eps.fillna(0)
        # for (attack, eps), group in data.groupby(['Attack', 'Eps']):
        for attack, group in data.groupby('Attack'):
            if attack == 'auth': continue

            # alabel = '{}-e{}'.format(attack, eps) if eps else attack
            alabel = attack

            pred = np.concatenate((group.pred.values, auths.pred.values))
            target = np.concatenate((group.target.values, auths.target.values))

            fpr, tpr, thr, eer_idx, eer_accuracy, auc = evaluate(pred, target)
            # plot_fn(fpr, tpr, label='{} (AUC={:4.3f})'.format(alabel, auc))

            broken_down_aucs = broken_down_aucs.append(pd.DataFrame({
                'Method': ('MLP' if params.mlp.values[0] else 'LSTM'),
                'Centroids': ('M' if 'medoid' in params.centroids.values[0] else 'C'),
                'Distance': ('cos' if params.distance.values[0] == 'cosine' else 'L_2'),
                'Attack': attack,
                'AUC': auc,
                'EERAcc': eer_accuracy
            }, index=[0]), ignore_index=True)

    # random line
    a = np.logspace(-4, 0)
    plot_fn(a, a, '--', label='random', c='k')

    aucs.append(0.5)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(10**-2.5, 1)
    plt.ylim(0, 1)

    plt.minorticks_on()
    plt.grid(which='minor', axis='y', linestyle='dashed', c=(0.9,) * 3)

    # legend sorted by AUC
    #handles, labels = plt.gca().get_legend_handles_labels()
    #aucs, labels, handles = zip(*sorted(zip(aucs, labels, handles), key=lambda t: t[0])[::-1])
    #plt.gca().legend(handles, labels) #, loc='lower right')
    plt.legend()
    plt.savefig(args.output)

    print('AUC')
    pt = broken_down_aucs.pivot_table(index=['Centroids', 'Method', 'Distance'], columns=['Attack'], values='AUC')
    pt = pt.reindex(['lbfgs', 'fgsm', 'iter', 'pgd', 'm-iter'], axis=1)
    pt = pt.reindex(['M', 'C'], axis=0, level=0)
    pt = pt.reindex(['cos', 'L_2'], axis=0, level=2)

    pt.columns = ['L-BFGS', 'FGSM', 'BIM', 'PGD', 'MI-FGSM']
    pt['Macro-AUC'] = pt.mean(axis=1)
    pt = pt.applymap('{:3.3f}'.format)
    print(pt)
    with open('auc.tex', 'w') as f:
        f.write(pt.to_latex())

    print('EER')
    pt = broken_down_aucs.pivot_table(index=['Centroids', 'Method', 'Distance'], columns=['Attack'], values='EERAcc')
    pt = pt.reindex(['lbfgs', 'fgsm', 'iter', 'pgd', 'm-iter'], axis=1)
    pt = pt.reindex(['M', 'C'], axis=0, level=0)
    pt = pt.reindex(['cos', 'L_2'], axis=0, level=2)
    pt.columns = ['L-BFGS', 'FGSM', 'BIM', 'PGD', 'MI-FGSM']
    pt['Macro-AUC'] = pt.mean(axis=1)
    pt = pt.applymap('{:3.3f}'.format)
    print(pt)
    with open('eer.tex', 'w') as f:
        f.write(pt.to_latex())
