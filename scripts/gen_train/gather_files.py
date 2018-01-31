from pathlib import Path

from functional import seq
from fn import _ as X, F
import pandas as pd
import numpy as np
import random
# noinspection PyUnresolvedReferences
from plumbum.cmd import ls, rm

RES_DIR = '/Users/qianws/jupyterNotebooks/leap-scd/res/'
TIMIT_DIR = Path(RES_DIR) / 'timit'
gather_wav_dir = TIMIT_DIR / 'gathered_wavs'
gather_wav_dir.mkdir(exist_ok=True)
gather_phn_dir = TIMIT_DIR / 'gathered_phns'
gather_phn_dir.mkdir(exist_ok=True)
lists_dir = Path(RES_DIR) / '../lists'


def gather_all_wav_phn_files():
    """ gather around all wav/phn files, for flatten path """

    # for fn in TIMIT_DIR.glob('**/*.wav'): # just run once by hand
    #     concat_name = '_'.join(fn.parts[-2:]).upper()[:-4] + '.wav'
    #     fn.rename(gather_wav_dir / concat_name)

    for fn in TIMIT_DIR.glob('**/*.phn'):  # just run once by hand
        concat_name = '_'.join(fn.parts[-2:]).upper()[:-4] + '.phn'
        fn.rename(gather_phn_dir / concat_name)


def split_into_train_val_test():
    """
    there are 6300 files in total, move them into corresponding sub dirs
    and sph2wav on the fly
    train:  5000
    val:    1000
    test:    300
    :return:
    """
    all_wav_files = list(gather_wav_dir.glob('*.wav'))
    np.random.shuffle(all_wav_files)

    from plumbum import local
    sph2pipe = local['/Users/qianws/ideaProjects/kaldi-gop-cpp/kaldi/tools/sph2pipe_v2.5/sph2pipe']['-f', 'wav']

    for dir_name, slc in zip(['train', 'test', 'val'], [(None, 5000), (5000, 6000), (6000, None)]):
        d = gather_wav_dir / dir_name
        d.mkdir(exist_ok=True)
        (seq(all_wav_files[slice(*slc)])
         # convert into wav @target dir
         .for_each(lambda f: sph2pipe(f.as_posix(), (d / f.name).as_posix()))
         )

    # remove raw sph files(with fake '.wav' suffix)
    seq(all_wav_files).for_each(lambda f: rm(f.as_posix()))


def sample_trainfile1_2_num():
    """create 3 list files: trainfile1, trainfile2, trainnumbers
    for use in wrapper_for_gen.wav
    ensure that trainfile1's speaker is different from trainfile2
    """
    s1 = seq((gather_wav_dir / 'train').glob('*.wav')).map(X.name).list() * 3
    s2 = s1.copy()
    random.shuffle(s2)

    df_join = pd.DataFrame(data={'fn1': s1, 'fn2': s2})
    df_join['spk1'] = df_join.fn1.apply(X[:5])
    df_join['spk2'] = df_join.fn2.apply(X[:5])
    df_join = (df_join.query('spk1 != spk2')
               .sample(frac=1, replace=False)
               )

    seq(df_join.fn1.values) \
        .to_file((lists_dir / 'trainfile1.list').as_posix(), delimiter='\n')  # CAUTION: delimiter
    seq(df_join.fn2.values) \
        .to_file((lists_dir / 'trainfile2.list').as_posix(), delimiter='\n')
    seq.range(df_join.shape[0]) \
        .to_file((lists_dir / 'trainnumbers.list').as_posix(), delimiter='\n')
    print()


def main():
    # gather_all_wav_phn_files()
    # split_into_train_val_test()
    sample_trainfile1_2_num()
    pass


main()
