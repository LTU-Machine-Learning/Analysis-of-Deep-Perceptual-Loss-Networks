import csv
from statistics import mean

from workspace_path import home_path

path = home_path / 'logs/perceptual_metric'

fields = [
    'architecture', 'extraction_layers', 'pretrained', 'frozen', 'variant', 
    'lpips_norm', 'metric_fun', 'no_channel_norm', 'version', 'test_loss_cnn', 
    'test_score_cnn', 'test_accuracy_cnn', 'test_loss_color',
    'test_score_color', 'test_accuracy_color', 'test_loss_deblur',
    'test_score_deblur', 'test_accuracy_deblur', 'test_loss_frameinterp',
    'test_score_frameinterp', 'test_accuracy_frameinterp',
    'test_loss_superres', 'test_score_superres', 'test_accuracy_superres',
    'test_loss_traditional', 'test_score_traditional', 
    'test_accuracy_traditional', 'test_jnd', 'test_loss_avg', 'test_score_avg',
    'test_accuracy_avg'									
]

replacements = [
    ('FeatureExtractor_', ''), ('True_', 'True-'), ('False_', 'False-'),
    ('baseline_', 'baseline-'), ('lin_', 'lin-'), ('scratch_', 'scratch-'),
    ('tune_', 'tune-'), ('lpips-norm_', 'lpips_norm-'), (' ', ''),
    ('no-channel-norm', 'no_channel_norm'), ('b1.0', 'spatial+sort'),
    ('bpure', ''), ('bmean', 'spatial+mean'), ('bsort', 'spatial+sort'),
    ('_no_', '-no_')
]

file = path / 'results.csv'

if not file.exists():
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(fields)

rows = {}
with open(file, newline='') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        rows['-'.join(row[:8])] = row[8:]

with open(file, 'a', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    for folder in path.iterdir():
        if not folder.is_dir():
            continue
        for version_folder in folder.iterdir():
            if not version_folder.is_dir():
                continue
            if not (version_folder / 'test_results.csv').exists():
                continue
            name = folder.name
            for x, y in replacements:
                name = name.replace(x,y)
            name = name.split('-')
            
            true_name = name[:4]
            true_name.append(name[6])
            true_name.append('yes' if 'lpips_norm' in name else 'no')
            funs = ['spatial', 'mean', 'sort', 'spatial+mean', 'spatial+sort']
            fun = [f for f in funs if f in name]
            if len(fun) == 0:
                true_name.append('spatial')
            elif len(fun) == 1:
                true_name.append(fun[0])
            else:
                raise RuntimeError(
                    f'Expected {folder.name} to contain max 1 of {funs}'
                )
            true_name.append('yes' if 'no_channel_norm' in name else 'no')
            true_name.append(version_folder.name[-1])

            assert(len(true_name) == 9)
            if '-'.join(true_name) in rows:
                continue

            with open(version_folder / 'test_results.csv') as read_file:
                reader = csv.reader(read_file, delimiter=' ')
                data = [row for row in reader]
                val = [row[1] for row in data]
                val.append(
                    mean([float(row[1]) for row in data if 'loss' in row[0]])
                )
                val.append(
                    mean([float(row[1]) for row in data if 'score' in row[0]])
                )
                val.append(mean(
                    [float(row[1]) for row in data if 'accuracy' in row[0]]
                ))
                writer.writerow(true_name + val)