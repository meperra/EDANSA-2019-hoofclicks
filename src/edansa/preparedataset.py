"""Process excell from megan and taxonomy.yaml to create dataset.

Exported from prepare_dataset notebook.
"""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
import yaml
import csv
from pprint import pprint
from copy import deepcopy
from collections import Counter
from datetime import datetime

from edansa import dataimport, taxoutils

# # %%
# #Parameters
# sample_count_limit = 25
# sample_length_limit = 10


# %%
def load_file_info2dataset(dataset_rows,
                           dataset_name_v='',
                           dataset_cache_folder='',
                           dataset_folder=None):
    """read path, len of megan labeled files from csv file, (lnength col.)
     store them in a dataimport.dataset, keys are gonna be sample file path 
    """
    # src_path = '/scratch/enis/data/nna/labeling/megan/AudioSamplesPerSite/'
    # ffp = src_path + 'meganLabeledFiles_wlenV1.txt'

    audio_dataset = dataimport.Dataset(
        dataset_rows,
        dataset_name_v=dataset_name_v,
        excerpt_len=excerpt_length,
        dataset_cache_folder=dataset_cache_folder,
        excell_names2code=excell_names2code,
        taxonomy_file_path=taxonomy_file_path,
        target_taxo=target_taxo,
    )

    audio_dataset.load_csv(dataset_rows,
                           dataset_name_v=dataset_name_v,
                           dataset_cache_folder=dataset_cache_folder,
                           dataset_folder=dataset_folder)

    return audio_dataset


# %%


def load_labeled_info(dataset_rows, audio_dataset, ignore_files=None):
    """Read labeled info from spreat sheet
        and remove samples with no audio file, also files given in ignore_files
    """
    if ignore_files is None:
        ignore_files = set()

    missing_audio_files = []
    for row in dataset_rows:
        if audio_dataset.get(row['Clip Path'], None) is None:
            missing_audio_files.append(row['Clip Path'])

    missing_audio_files = set(missing_audio_files)
    print((f'{len(missing_audio_files)} files are missing' +
           ' corresponding to excell entries'))

    dataset_rows_filtered = []
    for row in dataset_rows:
        if row['Clip Path'] not in ignore_files:
            if row['Clip Path'] not in missing_audio_files:
                dataset_rows_filtered.append(row)

    deleted_files = set()
    deleted_files.update(ignore_files)
    deleted_files.update(missing_audio_files)
    pprint((f'-> {len(deleted_files)} number of samples are DELETED due to ' +
            'ignore_files and missing_audio_files'))

    return dataset_rows_filtered, list(deleted_files)


# %%


def load_taxonomy2dataset(taxonomy_file_path, audio_dataset):
    # Store taxonomy information in the dataset.
    # Taxonomy file
    taxonomy_file_path = Path(taxonomy_file_path)
    with open(taxonomy_file_path) as f:
        taxonomy = yaml.load(f, Loader=yaml.FullLoader)

    t = taxoutils.Taxonomy(deepcopy(taxonomy))
    audio_dataset.taxonomy = t


# %%
def add_taxo_code2dataset(dataset_rows, audio_dataset, version='V2'):
    '''Go through rows of the excell and store taxonomy info into audio_dataset
        version: with V2, taxo codes are list
    '''

    for row in dataset_rows:
        taxonomy_codes = taxoutils.megan_excell_row2yaml_code(
            row, audio_dataset.excell_names2code, version=version)
        audio_sample = audio_dataset[row['Clip Path']]
        site_id = row['Site ID'].strip()
        audio_sample.location_id = site_id
        audio_sample.taxo_codes = taxonomy_codes


def del_samples_not_labeled(audio_dataset, dataset_rows):
    """ remove files from dataset that are not in the labeled list

        we have some audio files in the original folder but
        they are not samples but original big recordings etc
        remove anything that is not in the list

    """

    megan_data_sheet_file_names = {row['Clip Path'] for row in dataset_rows}

    to_be_deleted = []
    for key in audio_dataset:
        if key not in megan_data_sheet_file_names:
            to_be_deleted.append(key)
    for k in to_be_deleted:
        del audio_dataset[k]
    print(
        f'-> {len(to_be_deleted)} samples DELETED because they are not in the '
        + 'excell\n')
    return to_be_deleted


def del_samples_w_no_taxo(audio_dataset):
    """remove samples without taxonomy code from dataset

    if audio file not in the exell then they do not have taxo info
    """
    to_be_deleted = []
    for k, audio in audio_dataset.items():
        if audio.taxo_codes is None:
            to_be_deleted.append(k)
    for k in to_be_deleted:
        del audio_dataset[k]
    print(
        f'-> {len(to_be_deleted)} samples DELETED because they do not have the '
        + 'taxo info coming from excell\n')
    return to_be_deleted


def count_category_size(audio_dataset, sample_length_limit, version='V2'):
    """Go through rows of the excell and count category population
    """
    taxo_code_counter = audio_dataset.count_samples_per_taxo_code(
        sample_length_limit, version=version)

    return taxo_code_counter


# %%
def delete_samples_by_taxo_limit(taxo_code_counter, audio_dataset,
                                 taxo_count_limit):
    """find taxonomies with not enough data and delete all samples from taxo

    """
    taxonomy_no_enough_data = []
    print('-> classes that do not have enough data:\n[REMOVED!]')
    for k, v in taxo_code_counter.items():
        if v < taxo_count_limit:
            print(audio_dataset.taxonomy.edges[k], v)
            taxonomy_no_enough_data.append(k)

    print('\n-> classes that have enough data:')
    for k, v in taxo_code_counter.items():
        if v >= taxo_count_limit:
            print(audio_dataset.taxonomy.edges[k], v)

    #  DELETE taxonomy with not enough data
    samples_2_delete = []
    for k, v in audio_dataset.items():
        for taxo_code in v.taxo_codes:
            if taxo_code in taxonomy_no_enough_data:
                samples_2_delete.append(k)
                break

    for k in samples_2_delete:
        del audio_dataset[k]

    pprint(
        f'-> {len(samples_2_delete)} number of samples are deleted because ' +
        'their taxonomy category does not have enough data.')

    return samples_2_delete


# %%


def delete_samples_by_length_limit(audio_dataset, sample_length_limit):
    """find samples that are not long enough and delete samples from dataset

    """
    sample_not_long_enough = []
    print('-> classes that do not have enough data\nwill be REMOVED!')
    for k, v in audio_dataset.items():
        if v.length < sample_length_limit:
            sample_not_long_enough.append(k)

    #  DELETE samples with not enough data
    for k in sample_not_long_enough:
        del audio_dataset[k]

    print(f'-> {len(sample_not_long_enough)} number of samples are deleted ' +
          'because their length is not long enough.')

    return sample_not_long_enough


#%%
def load_csv(csv_path):
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        reader = list(reader)
        reader_strip = []
        for row in reader:
            row = {r: row[r].strip() for r in row}
            reader_strip.append(row)
        reader = reader_strip.copy()
    return reader


def run(
    dataset_csv_path,
    taxonomy_file_path,
    ignore_files,
    excerpt_length,
    sample_length_limit,
    taxo_count_limit,
    excell_names2code=None,
    dataset_name_v='',
    dataset_cache_folder='',
    version='V2',
    load_clipping=True,
    dataset_folder=None,
    target_taxo=None,
):
    megan_data_sheet = load_csv(dataset_csv_path)
    audio_dataset = dataimport.Dataset(
        megan_data_sheet,
        dataset_name_v=dataset_name_v,
        excerpt_len=excerpt_length,
        dataset_cache_folder=dataset_cache_folder,
        excell_names2code=excell_names2code,
        dataset_folder=dataset_folder,
        taxonomy_file_path=taxonomy_file_path,
        target_taxo=target_taxo,
    )
    megan_data_sheet, deleted_files = load_labeled_info(
        megan_data_sheet, audio_dataset, ignore_files=ignore_files)

    # audio_dataset.excerpt_length = excerpt_length
    # audio_dataset.excell_names2code = excell_names2code  # type: ignore

    # load_taxonomy2dataset(taxonomy_file_path, audio_dataset)
    # add_taxo_code2dataset(megan_data_sheet, audio_dataset, version=version)

    deleted_samples_not_labeled = del_samples_not_labeled(
        audio_dataset, megan_data_sheet)
    deleted_samples_w_no_taxo = del_samples_w_no_taxo(audio_dataset)
    taxo_code_counter = count_category_size(audio_dataset,
                                            sample_length_limit,
                                            version=version)

    # Apply Limits taxo_count_limit and sample_length_limit
    small_taxo_deleted = delete_samples_by_taxo_limit(taxo_code_counter,
                                                      audio_dataset,
                                                      taxo_count_limit)
    sample_not_long_enough = delete_samples_by_length_limit(
        audio_dataset, sample_length_limit)
    if load_clipping:
        audio_dataset.update_samples_w_clipping_info()
    deleted_files = (deleted_files + small_taxo_deleted +
                     sample_not_long_enough + deleted_samples_not_labeled +
                     deleted_samples_w_no_taxo)
    return audio_dataset, deleted_files
