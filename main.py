import os
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
import numpy as np

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
print('PyTorch version:', torch.__version__)
print('Seed set to 0')

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)

parser.add_argument('--dataset', type=str, required=True) #default='ECG_data')
parser.add_argument('--window_size', type=int, required=True) #default=12)
parser.add_argument('--horizon', type=int, required=True) #default=3)

parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=-1)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--bottom_level_only', type=bool)


args = parser.parse_args()
from pprint import pprint
print('Training configs:')
pprint(vars(args))
# Get script path
script_path = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(script_path, 'dataset', args.dataset + '.csv')
result_train_file = os.path.join(script_path, 'output', args.dataset, 'train')
result_test_file = os.path.join(script_path, 'output', args.dataset, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
if args.dataset.startswith('hier-'):
    from datasetsforecast.hierarchical import HierarchicalData
    Y_df, S_df, tags = HierarchicalData.load('./data', args.dataset.split('-')[1])
        # Long format data (T, N); Aggregation matrix; Tags
    # Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    Y_wide = Y_df.pivot(index='ds', columns='unique_id', values='y')
    assert Y_wide.isna().sum().sum() == 0, 'Missing values in the data'

    if args.bottom_level_only: # Get the bottom level
        print("Using the bottom level data")
        # cnt_lvl_map = { len(tag): lvl for lvl, tag in tags.items() }
        # bottom_lvl_names = cnt_lvl_map[max(cnt_lvl_map.keys())]
        # Y_wide_bottom = Y_wide[tags[bottom_lvl_names]]
        Y_wide_bottom = Y_wide[S_df.columns]
        data = Y_wide_bottom.values
        A = None
    else:
        print("Using all levels of data")
        data = Y_wide.values
        # Convert aggregation matrix to adjacency matrix
        n_total, n_bottom = S_df.shape
        n_agg = n_total - n_bottom
        # Get the first n_agg rows
        S_agg = S_df.iloc[:n_agg, :]
        # Create adjacency matrix
        A = np.hstack([
            np.vstack([np.eye(n_agg), -S_agg.values.T]),
            S_df.values
        ])
        # A = A - np.eye(n_total)
else:
    data = pd.read_csv(data_file).values  # Wide format: (T, N)
    A = None


# split data
train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
test_ratio = 1 - train_ratio - valid_ratio
train_data = data[:int(train_ratio * len(data))]
valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
test_data = data[int((train_ratio + valid_ratio) * len(data)):]

torch.manual_seed(0)
if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(train_data, valid_data, args, result_train_file, prior_matrix=A)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60:.2f} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60:.3f} minutes')
    pprint(vars(args))
    print('done')