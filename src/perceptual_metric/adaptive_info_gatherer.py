import csv
from statistics import mean

from workspace_path import home_path

path = home_path / 'logs/adaptive_metric'

param_fields = [
    'train_set', 'test_sets', 'augment_order', 'architecture',
    'extraction_layers', 'variant', 'lpips_norm', 'metric_fun', 'sync_loss',
    'no_channel_norm', 'version'
]
result_fields = [
    'test_loss_cnn', 'test_score_cnn', 'test_accuracy_cnn',
    'test_loss_color', 'test_score_color', 'test_accuracy_color',
    'test_loss_deblur', 'test_score_deblur', 'test_accuracy_deblur',
    'test_loss_frameinterp', 'test_score_frameinterp',
    'test_accuracy_frameinterp',
    'test_loss_superres', 'test_score_superres', 'test_accuracy_superres',
    'test_loss_traditional', 'test_score_traditional', 
    'test_accuracy_traditional',
    'test_loss_SVHN', 'test_score_SVHN', 'test_accuracy_SVHN',
    'test_loss_STL10', 'test_score_STL10', 'test_accuracy_STL10',
    'test_jnd',
    'test_loss_2afc', 'test_score_2afc', 'test_accuracy_2afc',
    'test_loss_order', 'test_score_order', 'test_accuracy_order',								
]

replacements = [
    ('zoom_in', 'zoom'), ('FeatureExtractor_', ''), ('-True', ''),
    ('-False', ''), ('no-channel-norm', 'no'), ('channel-norm', 'yes'),
    ('imagenet-norm', 'no'), ('lpips-norm', 'yes'), ('net1_1', 'net1.1'),
    ('--1','-until'), ('sync-loss-', ''), ('-[', '_['), ('[', ''), (']', ''),
    ('-', ', ')
]

file = path / 'results.csv'

if not file.exists():
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(param_fields + result_fields)

rows = {}
with open(file, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        rows['_'.join(row[:len(param_fields)])] = row[len(param_fields):]

with open(file, 'a', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for folder in path.iterdir():
        if not folder.is_dir():
            continue
        for version_folder in folder.iterdir():
            if not version_folder.is_dir():
                continue
            if not (version_folder / 'test_results.csv').exists():
                continue
            
            with open(version_folder / 'test_results.csv') as read_file:
                name = folder.name
                for x, y in replacements:
                    name = name.replace(x,y)  
                if '-'.join(name) in rows:
                    continue
                name = name.split('_')
                name.append(version_folder.stem[-1])

                reader = csv.reader(read_file, delimiter=' ')
                data = [row for row in reader]
                val = [row[1] for row in data]
                val.append(
                    mean([
                        float(row[1])
                        for row in data
                        if 'loss' in row[0] and not 'S' in row[0]
                    ])
                )
                val.append(
                    mean([
                        float(row[1])
                        for row in data
                        if 'score' in row[0] and not 'S' in row[0]
                    ])
                )
                val.append(
                    mean([
                        float(row[1])
                        for row in data
                        if 'accuracy' in row[0] and not 'S' in row[0]
                    ])
                )
                val.append(
                    mean([
                        float(row[1])
                        for row in data
                        if 'loss' in row[0] and 'S' in row[0]
                    ])
                )
                val.append(
                    mean([
                        float(row[1])
                        for row in data
                        if 'score' in row[0] and 'S' in row[0]
                    ])
                )
                val.append(
                    mean([
                        float(row[1])
                        for row in data
                        if 'accuracy' in row[0] and 'S' in row[0]
                    ])
                )

                writer.writerow(name + val)