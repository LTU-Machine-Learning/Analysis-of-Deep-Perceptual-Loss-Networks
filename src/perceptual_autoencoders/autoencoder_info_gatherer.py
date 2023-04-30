import csv

from workspace_path import home_path

path = home_path / 'checkpoints/perceptual_autoencoders'

fields = [
    'dataset', 'input_size_0', 'input_size_1', 'ae_epochs', 'encoder', 
    'z_dim', 'gamma', 'loss_net', 'beta_factor', 'ae_version', 'predictor',
    'act_func', 'out_func', 'predictor_epochs', 'predictor_version',
    'val_loss', 'val_l1_distance', 'val_l2_distance', 'val_accuracy',
    'test_loss', 'test_l1_distance', 'test_l2_distance', 'test_accuracy'
]

file = path / 'results.csv'
 
if not file.exists():
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(fields)

rows = {}
with open(file, newline='') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        rows['_'.join(row[:-8])] = row[-8:]

with open(file, 'a', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for encoder_dir in path.iterdir():
        if not encoder_dir.is_dir():
            continue
        for encoder_version_dir in encoder_dir.iterdir():
            if not encoder_version_dir.is_dir():
                continue
            encoder_version = encoder_version_dir.name[-1]
            for predictor_dir in encoder_version_dir.iterdir():
                if not predictor_dir.is_dir():
                    continue
                for predictor_version_dir in predictor_dir.iterdir():
                    if not predictor_version_dir.is_dir():
                        continue
                    row_name = (
                        predictor_dir.name+f'_{predictor_version_dir.name[-1]}'
                    )
                    replacements = [
                        ('FeatureExtractor_',''), ('_bn','-bn'),
                        ('wide_','wide-'), ('_v2','-v2'), ('_v3','-v3'),
                        ('_x_','-x-'), ('_x','-x'), ('_y_','-y-'),
                        ('x1_','x1-'), ('x2_','x2-'),('_2gf','-2gf'),
                        ('_6gf','-6gf'), ('t0_','t0-'), ('t1_','t1-'),
                        ('t_b','t-b'), ('_32x','-32x')
                    ]
                    for x, y in replacements:
                        row_name = row_name.replace(x,y)
                    row_name = row_name.split('_')
                    if len(row_name) == len(fields)-10:
                        row_name.insert(8, 'None')
                    row_name.insert(9, encoder_version)
                    assert len(row_name) == len(fields)-8
                    row_name = '_'.join(row_name)
                    if row_name in rows:
                        continue
                    if (predictor_version_dir / 'results.csv').exists():
                        with open(
                            predictor_version_dir / 'results.csv', newline=''
                        ) as read_file:
                            reader = csv.reader(read_file, delimiter=' ')
                            for r in reader:
                                if len(r)==8 and r[0] != 'val_loss':
                                    row = row_name.split('_') + r
                                    writer.writerow(row)
                                    break
