import os
import shutil
import pandas as pd
from random import sample


def copy_files(from_dir, save_dir, file_ids):
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    for f_id in file_ids:
        f = from_dir + '/' + f_id + '.txt'
        shutil.copyfile(f, save_dir + '/' + f_id + '.txt')


def cut_file(from_path, to_path):
    shutil.copyfile(from_path, to_path)
    os.remove(from_path)


def move_files(from_dir, save_dir, file_ids):
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    for f_id in file_ids:
        f = from_dir + '/' + f_id + '.txt'
        cut_file(f, save_dir + '/' + f_id + '.txt')


def get_gold(save_dir, gold, file_ids):
    file_gold = gold[gold['id'].isin(file_ids)]
    file_gold.to_csv(save_dir, index=False)


print("Step 0: split data into clusters.")
file_dir = './data'
clusters = pd.read_csv('clusters.csv', header=0)
gold = pd.read_csv(file_dir + '/train.csv', header=0)

for cluster in range(0, 15):
    cluster_ids = clusters.loc[clusters['cluster'] == cluster]['id'].tolist()
    copy_files(file_dir + '/train', file_dir + '/' + str(cluster), cluster_ids)
    get_gold(file_dir + '/' + str(cluster) + '/' + 'gold.csv', gold, cluster_ids)

print("Step 1: SamePrompt - split each clusters into 90% Train, 10% Validation and 10% Test")
file_dir = './data'
split = pd.read_csv("DatenSplitten.txt", header=0, sep='\t')
clusters_tests = {}
clusters_validate = {}
clusters_train = {}
for cluster in range(0, 15):
    cluster_ids = clusters.loc[clusters['cluster'] == cluster]['id'].tolist()
    # Test
    tests_ids = sample(cluster_ids, k=split['Test'][cluster])
    clusters_tests[cluster] = tests_ids
    save_dir = file_dir + '/same_prompt/' + str(cluster) + '/' + 'Test' + '/'
    copy_files(file_dir + '/' + str(cluster), save_dir, tests_ids)
    get_gold(save_dir + 'test.csv', gold, tests_ids)

    # Validate
    train80 = list(set(cluster_ids) - set(tests_ids))
    validate_ids = sample(train80, k=split['Validation'][cluster])
    clusters_validate[cluster] = validate_ids
    save_dir = file_dir + '/same_prompt/' + str(cluster) + '/' + 'Validation' + '/'
    copy_files(file_dir + '/' + str(cluster), save_dir, validate_ids)
    get_gold(save_dir + 'validate.csv', gold, validate_ids)

    # Train
    train_ids = list(set(train80) - set(validate_ids))
    clusters_train[cluster] = train_ids
    save_dir = file_dir + '/same_prompt/' + str(cluster) + '/' + 'Train' + '/'
    copy_files(file_dir + '/' + str(cluster), save_dir, train_ids)
    get_gold(save_dir + 'train.csv', gold, train_ids)

print("Step 2: OtherPrompts - 12 of the 14 other prompts for training and 2 for validation")
for cluster in range(0, 15):
    train_ids = []
    for train_cluster in range(cluster + 3, cluster + 14):
        if train_cluster > 14:
            train_cluster = train_cluster - 14
        train_ids.extend(clusters.loc[clusters['cluster'] == train_cluster]['id'].tolist())
    save_dir = file_dir + '/other_prompts/' + str(cluster) + '/' + 'Train' + '/'
    copy_files(file_dir + '/train', save_dir, train_ids)
    get_gold(save_dir + 'train.csv', gold, train_ids)

    validate_ids = []
    for validate_cluster in range(cluster + 1, cluster + 3):
        if validate_cluster > 14:
            validate_cluster = validate_cluster - 14
        validate_ids.extend(clusters.loc[clusters['cluster'] == validate_cluster]['id'].tolist())
    save_dir = file_dir + '/other_prompts/' + str(cluster) + '/' + 'Validation' + '/'
    copy_files(file_dir + '/train', save_dir, validate_ids)
    get_gold(save_dir + 'validate.csv', gold, validate_ids)

print("Step 3: AllPrompts - combine all the training and validation data in cluster 0-14")
all_train_ids = []
all_validate_ids = []
for cluster in range(0, 15):
    all_train_ids.extend(clusters_train.get(cluster))
    all_validate_ids.extend(clusters_validate.get(cluster))
save_dir = file_dir + '/all_prompts/Train/'
copy_files(file_dir + '/train', save_dir, all_train_ids)
get_gold(save_dir + 'train.csv', gold, all_train_ids)
save_dir = file_dir + '/all_prompts/Validation/'
copy_files(file_dir + '/train', save_dir, all_validate_ids)
get_gold(save_dir + 'validate.csv', gold, all_validate_ids)

print(
    "Step 4: OtherPromptsSmall - sample down OtherPrompts to the same average amount of training data as used in the SamePrompt condition")
other_setting = [[0, 0, 0, 49, 96, 98, 58, 48, 60, 73, 49, 69, 79, 101, 53],
                 [63, 0, 0, 0, 94, 96, 57, 47, 58, 72, 48, 67, 78, 99, 52],
                 [66, 58, 0, 0, 0, 101, 60, 49, 61, 75, 51, 71, 82, 104, 54],
                 [64, 56, 126, 0, 0, 0, 58, 47, 59, 73, 49, 68, 79, 100, 52],
                 [65, 57, 127, 49, 0, 0, 0, 48, 60, 73, 50, 69, 80, 101, 53],
                 [61, 54, 120, 46, 91, 0, 0, 0, 57, 69, 47, 65, 75, 96, 50],
                 [59, 51, 115, 44, 88, 89, 0, 0, 0, 66, 45, 62, 72, 92, 48],
                 [60, 52, 117, 45, 89, 91, 54, 0, 0, 0, 46, 63, 73, 93, 49],
                 [60, 52, 117, 45, 89, 91, 54, 44, 0, 0, 0, 64, 74, 93, 49],
                 [60, 53, 118, 46, 90, 92, 55, 45, 56, 0, 0, 0, 74, 94, 49],
                 [61, 53, 119, 46, 91, 92, 55, 45, 56, 69, 0, 0, 0, 95, 50],
                 [64, 57, 127, 49, 96, 98, 58, 48, 60, 73, 49, 0, 0, 0, 53],
                 [63, 56, 124, 48, 95, 96, 57, 47, 58, 72, 48, 67, 0, 0, 0],
                 [0, 55, 122, 47, 93, 95, 56, 46, 57, 71, 48, 66, 77, 0, 0],
                 [0, 0, 116, 45, 88, 90, 53, 44, 55, 67, 45, 63, 73, 93, 0]]

for cluster in range(0, 15):
    setting = other_setting[cluster]
    other_train_in_cluster = []
    for c in range(0, 15):
        candidate = clusters_train.get(cluster)
        other_train_in_cluster.extend(sample(candidate, k=setting[c]))
    save_dir = file_dir + '/other_prompts_small/' + str(cluster) + 'Train/'
    copy_files(file_dir + '/train', save_dir, other_train_in_cluster)
    get_gold(save_dir + 'train.csv', gold, other_train_in_cluster)

print(
    "Step 5: AllPromptsSmall - sample down AllPrompts to the same average amount of training data as used in the SamePrompt condition")
all_setting = [50, 44, 97, 38, 74, 75, 45, 37, 46, 56, 38, 53, 61, 78, 41]
all_train_small = []
for cluster in range(0, 15):
    candidate = clusters_train.get(cluster)
    all_train_small.extend(sample(candidate, k=all_setting[cluster]))

save_dir = file_dir + '/all_prompts_small/Train/'
copy_files(file_dir + '/train', save_dir, all_train_small)
get_gold(save_dir + 'train.csv', gold, all_train_small)
