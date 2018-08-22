import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

sns.set('paper', 'whitegrid', font='serif', font_scale=1, rc={
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
    dist = fpr**2 + (1-tpr)**2
    best = np.argmin(dist)
    best = fpr[best], tpr[best], thr[best], auc

    return fpr, tpr, thr, eer_idx, eer_accuracy, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Adversarial Detectors ROC')
    # DATA PARAMS
    parser.add_argument('data', help='Data to plot')
    parser.add_argument('-a', '--adversarial-info', default='adversarial.csv', help='CSV with adversarial info')
    parser.add_argument('-o', '--output', help='Output plot file')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.data), 'roc.pdf')

    data = pd.read_csv(args.data, index_col=0)
    adv = pd.read_csv(args.adversarial_info)
    # adv['Image'] = adv['Path'].map(os.path.basename)
    
    data = data.join(adv.set_index('Path'), on='image')
    data.Attack = data.Attack.fillna('auth')

    # plot_fn = plt.loglog
    plot_fn = plt.plot
    
    with PdfPages(args.output) as pdf:
        plt.figure(figsize=(5, 5))
        # random line
        a = np.logspace(-4, 0)
        plot_fn(a, a, '--', label='random')
        
        '''
        # global ROC
        fpr, tpr, thr, eer_idx, eer_accuracy, auc = evaluate(data.pred, data.target)
        print('Global EER Accuracy: {:3.2%}'.format(eer_accuracy))
        print('EER-TPR: {:3.2%}'.format(tpr[eer_idx]))
        plot_fn(fpr, tpr, label='global (AUC={:4.3f})'.format(auc), lw=2, c='k')
        # plt.scatter(fpr[eer_idx], tpr[eer_idx])
        '''
        
        labels = {
            'fgsm': 'FGSM',
            'iter': 'BIM',
            'lbfgs': 'L-BFGS',
            'm-iter': 'MI-FGSM',
            'pgd': 'PGD'
        }
        
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
            plot_fn(fpr, tpr, label='{} (AUC={:4.3f})'.format(labels[alabel], auc))
            
        '''
        for attack in data.Attack.unique():
            if attack == 'auth': continue
            group = data[data['Attack'].isin(('auth', attack))]
            fpr, tpr, thr, eer_idx, eer_accuracy, auc = evaluate(group.pred, group.target)
            plt.plot(fpr, tpr, label='{} (AUC={:4.3f})'.format(attack, auc))
        '''
        plt.legend()
        plt.axis('equal')
        # plt.xlim(0, 1)
        plt.xlim(10**-2.3, 1)
        plt.ylim(0, 1)
        plt.minorticks_on()
        plt.grid(which='minor', axis='both', linestyle='dashed', c=(0.9,) * 3)
        pdf.savefig()
        plt.close()

        sys.exit(0)

        plot_fn = plt.semilogx
        plt.figure(figsize=(5, 5))
        
        for attack, group in data.groupby('Attack'):
            if attack == 'auth': continue
            if attack == 'lbfgs': continue
            
            # alabel = '{}-e{}'.format(attack, eps) if eps else attack
            alabel = attack

            pred = np.concatenate((group.pred.values, auths.pred.values))
            target = np.concatenate((group.target.values, auths.target.values))
            
            fpr, tpr, thr, eer_idx, eer_accuracy, auc = evaluate(pred, target)
            plot_fn(fpr, tpr, label='{} (AUC={:4.3f})'.format(alabel, auc))
            
        '''
        for attack in data.Attack.unique():
            if attack == 'auth': continue
            group = data[data['Attack'].isin(('auth', attack))]
            fpr, tpr, thr, eer_idx, eer_accuracy, auc = evaluate(group.pred, group.target)
            plt.plot(fpr, tpr, label='{} (AUC={:4.3f})'.format(attack, auc))
        '''
        plt.legend()
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0.7, 1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        pdf.savefig()
        plt.close()

        for attack, group in data.groupby('Attack'):
            # print(attack, group)
            # fpr, tpr, thr = metrics.roc_curve(group.target, group.pred)
            # auc = metrics.auc(fpr, tpr)
            # plt.plot(fpr, tpr, label='{} (AUC={:4.3f})'.format(attack, auc))
            # import pdb; pdb.set_trace()
            scores = group.pred.sort_values().values
            distr = np.arange(len(scores)) / len(scores)
            metric = 'FNR'
            if attack == 'auth':
                distr = 1 - distr
                metric = 'FPR'
            plot_fn(scores, distr, label='{} ({})'.format(attack, metric))
        
        plt.legend()
        pdf.savefig()
        
        '''
        eer_accuracy, eer_thr, auc = evaluate(kept['pred'], kept['modified'])
        kept['eer_pred'] = kept['pred'] > eer_thr

        accs = kept.groupby('group').apply(lambda x: '{:3.2%}'.format(metrics.accuracy_score(x['eer_pred'], x['modified'])))
        print(accs)
        
        # pivot = kept[['modified', 'type', 'eer_pred']]
        pivot = pd.pivot_table(kept, values='pred', index=['modified', 'type'], columns=['eer_pred'], aggfunc='count')
        pivot.columns = ['Authenthic', 'Advesarial']
        pivot['Total'] = pivot['Authenthic'] + pivot['Advesarial']
        print(pivot)
        
        colors = np.array(('r', 'b'))
        
        n_samples = len(y)
        plt.scatter(np.arange(n_samples), y_hat, color=colors[y], marker='.')
        plt.savefig('plot.pdf')
        '''

