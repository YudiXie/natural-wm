from analysis.analysis_plot import performance_plot
from configs import experiments

__all__ = [
    'classification_pretrain_new_analysis',
    'classification_pretrain_mnist_analysis',
]

def classification_pretrain_new_analysis():
    cfg_df = experiments.classification_pretrain_new()
    performance_plot.bar_2par(cfg_df,
                              x_label='CNN width',
                              y_label='Classification Accuracy',
                              exp_name='classification_pretrain_new',
                              fig_name='cifar10_classification_acc',
                              fig_title='CIFAR-10 Classification Accuracy',
                              perf_key='TestAcc',
                              ylim=None,
                              hline=False,
                              bar_label=False)


def classification_pretrain_mnist_analysis():
    cfg_df = experiments.classification_pretrain_mnist()
    performance_plot.bar_2par(cfg_df,
                              x_label='CNN width',
                              y_label='Classification Accuracy',
                              exp_name='classification_pretrain_new',
                              fig_name='mnist_classification_acc',
                              fig_title='MNIST Classification Accuracy',
                              perf_key='TestAcc',
                              ylim=None,
                              hline=False,
                              bar_label=False)

