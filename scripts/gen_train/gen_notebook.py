# The idea is to just generate the wav file and the ground truth for speaker change detection.
import re
import os
import logging

import numpy as np
import scipy
import scipy.io.wavfile as wav
from functional import seq
from fn import _ as X, F

import htkmfc as htk

SAMPLE_RATE = 16000

#### DO NOT CHANGE ####
# base='/home/neerajs/work/NEW_REGIME/WAV/'
# base = '/home/siddharthm/'
clean = 'clean_wav/'
rev = 'rev_wav/'
rev_noise = 'rev_noise_wav/'
rev_inaud = 'rev_inaud_wav/'
# phndir = '/home/siddharthm/TIMIT/phones/'
phndir = '/Users/qianws/jupyterNotebooks/leap-scd/res/timit/gathered_phns/'
choices = [clean, rev, rev_noise, rev_inaud]
#### ------------- ####

# wavesavdir = '/home/siddharthm/scd/wavnew/train/'  # Save the generated wave file in this dir
wavesavdir = '/Users/qianws/jupyterNotebooks/leap-scd/res/wavnew/train/'  # Save the generated wave file in this dir
# over_addr = '/home/siddharthm/scd/vad/10/train/'  # labels in the directory
over_addr = '/Users/qianws/jupyterNotebooks/leap-scd/res/flabel/train/'  # labels in the directory

common_save = 'train_combinations'
### SOME VARIABLE DEFINITIONS ###
ratio_sc = 0.10
ratio_sil = 0.70
time_decision = 600  # in milliseconds
decision_samples = time_decision * 16  # Assuming 16KHz sampling rate
silence_samples = 0.00 * decision_samples  # The number of samples to be inserted[as silence]


###

# We are constructing ground truth from Phone files.
# There are many ways in which we can generate speaker change files. Right now, we are generating by concatenating the two files.

def data_saver(data):
    """ write (file1,file2) to common_save('train_combinations')
    :param data:
    :return:
    """
    # os.chdir('/home/siddharthm/scd/scores')
    os.chdir('/Users/qianws/jupyterNotebooks/leap-scd/res/scores')
    f = open(common_save + '.list', 'a+')
    f.write('\n')
    f.write(str(data))
    f.close()


def gen_func(file1, file2, input_index):
    """ merge 2 wav files, generate train data """
    logging.info('---------- %s, %s, %s', file1, file2, input_index)
    data_saver(str(file1) + "," + str(file2))

    file1_mod = file1.replace('_SX', '_S').replace('_SI', '_S')
    file2_mod = file1.replace('_SX', '_S').replace('_SI', '_S')

    # Fetching the base directory of the working files
    wav_addr1 = '/Users/qianws/jupyterNotebooks/leap-scd/res/timit/gathered_wavs/train/'
    wav_addr2 = wav_addr1

    # reading the two wav files and overlapping them
    wav_file1_mod = wav_addr1 + file1_mod + '.wav'
    wav_file1 = wav_addr1 + file1 + '.wav'
    wav_file2_mod = wav_addr2 + file2_mod + '.wav'
    wav_file2 = wav_addr1 + file2 + '.wav'

    phnFile1 = phndir + file1 + '.phn'
    phnFile2 = phndir + file2 + '.phn'

    #### Check if the first line begins with 0 for h#
    #### find silence parts of each file
    REGEX_SIL = re.compile(r'.*h#|epi|sil')

    def find_sil_in_phn_file(phn_file_path):
        with open(phn_file_path) as f:
            lines = f.readlines()
        res = (seq(lines)
               .map(str.rstrip)
               .filter(REGEX_SIL.match)
               .map(lambda ln: tuple(int(s) // 160 for s in ln.split(' ')[:2]))
               .to_list()
               )
        return res

    nsFile1, nsFile2 = seq([phnFile1, phnFile2]).map(find_sil_in_phn_file).to_list()

    # # Wavfile.read returns the sampling rate and the read data. The sampling rate is assumed to be 16KHz for our purposes.
    f = F(wav.read) >> X[1] >> (lambda a: a.reshape(1, -1).astype(float))  # X[1], drop the sample_rate
    a2, b2 = seq((wav_file1, wav_file2)).map(f)

    #### FRAME LEVEL MANIPULATIONS FOR CREATING OVERLAP LABELS ####
    nFrames1 = int(a2.shape[1] / 160)  # Number of frames that are possible from File one
    nFrames2 = int(b2.shape[1] / 160)  # Number of frames from File 2

    stFrame2 = nFrames1
    totalFrame = nFrames1 + nFrames2 - 1

    # labeling where are the frames from: 1*nFrames1 ... 0*nFrames, except for the silence
    labelFrames1 = np.ones((totalFrame,), dtype=np.int)
    # assign silence to 0
    for s, e in nsFile1: labelFrames1[s:e] = 0
    # for i in range(len(nsFile1)):
    #     labelFrames1[nsFile1[i][0]:nsFile1[i][1]] = 0
    labelFrames1[nFrames1:] = 0

    # 0*nFrames1 ... 2*nFrames2
    labelFrames2 = np.full((totalFrame,), 2,
                           dtype=np.int)  # Giving different identity to second speaker file[By using 2]
    labelFrames2[:stFrame2] = 0  # setting the left part
    for s, e in nsFile2:
        labelFrames2[s + stFrame2:e + stFrame2] = 0

    ### GENERATING THE FINAL LABEL FILE ###
    # terribly written, do not try to understand every line
    # just know that
    # * `out` is the concated wav data with silence trimed
    # * `flabels` is the 0/1/2 labels per 600ms, written in HTK format
    lastindex = np.where(labelFrames1 == 1)[0][-1]
    firstindex = np.where(labelFrames2 == 2)[0][0]
    silence_part = np.zeros((int(silence_samples / 160),), dtype=np.int)
    silence_actual_wav = np.zeros((int(silence_samples),))
    # print "Number of frames in File 1",nFrames1
    # print "Last non zero File1: ",lastindex,",First File 2: ",firstindex,",Length of complete vector: ",labelFrames2.shape
    labelpart1 = np.hstack((labelFrames1[0:lastindex + 1], silence_part))
    labelpart2 = np.hstack((labelFrames2[firstindex:]))  # silence part was not extraneous, was coming twice
    labelFrames = np.hstack((labelpart1, labelpart2))
    # print "Length of the labelFRames vector: ",labelFrames.shape
    start = 0
    iterator = 0
    skip_entries = int(decision_samples / 160)
    # print "Skip entries", skip_entries #By skip entries we mean the entries to be skipped in the label vector
    end = start + skip_entries
    ### GENERATING THE ACTUAL WAVE FILE ###
    # print "Actual samples from silence region: ",silence_actual_wav.shape[0]
    # print "Samples from First file: ",160*(lastindex+1)
    samplestart = 160 * labelFrames.shape[0] - (silence_actual_wav.shape[0] + 160 * (lastindex + 1))
    out = np.hstack((a2[0, :160 * (lastindex + 1)], silence_actual_wav, b2[0,
                                                                        -samplestart:-1]))  # Actually creating the numpy array which has the overlap and single speaker speech segments
    # print "Out wav file: ", out.shape
    flabels = []
    count, flag, count_TWO = 0, 0, 0
    # 2 for the silence class, 1 for speaker change frame, 0 for no speaker change frame
    while end < len(labelFrames):
        # Getting the vector ready
        aconsider = labelFrames[start:end]
        # print "Length of samples under consideration: ",160*len(aconsider)
        # Some definitions for further calculations
        count_zero = len(np.where(aconsider == 0)[0])
        count_one = len(np.where(aconsider == 1)[0])
        count_two = len(np.where(aconsider == 2)[0])
        # Decision section
        dec = -1
        if count_zero * 160 > int(ratio_sil * decision_samples):
            dec = 2
            count_TWO += 1
        elif min(count_one, count_two) * 160 > int(ratio_sc * decision_samples):
            dec = 1
            count += 1
        else:
            dec = 0
            flag += 1
        flabels.append(dec)
        iterator += 1
        start += 1
        end = skip_entries + start

    ### Setting the labels and the output ###
    out = out.astype(np.int16)
    out = np.reshape(out,
                     (out.shape[0], 1))  # Reshaping it to form the vector which is required to be written in wav file

    flabels = np.array(labelFrames)
    flabels = np.reshape(flabels, (1, flabels.shape[0]))
    ### SAVING THE STUFF SECTION ###
    scipy.io.wavfile.write(wavesavdir + file1 + '-' + file2 + '-' + str(input_index) + '.wav', SAMPLE_RATE, out)
    writer = htk.open(over_addr + file1 + '-' + file2 + '-' + str(input_index) + '.htk', mode='w',
                      veclen=max(flabels.shape))
    writer.writeall(flabels)


### TRY CALLS[Actual use with wrapper] ###
# gen_func('FCJF0_SX397', 'MJMM0_SX445', 15775)
# gen_func('FTBW0_X85','FTLG0_I1743',2)
pass
### ------- ###
