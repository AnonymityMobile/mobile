import datetime
import time
import os
from multiprocessing import Pool

from down_APK import get_sha256
from apk_single_decompile import apk_decompile, get_string_dex

from keywords_identity import main_keywords_identity

current_path = '/Users/Desktop/mobile/DLIdentifier'


def main_single(sha256_list):
    finished = 0
    for sha256_str in sha256_list:
        print('==down apk start==')
        down_command = 'curl -O --remote-header-name -G -d apikey=8c08e1e623110c600186098a11ba882a7e323ad32b71868510c971a504eca3f9 -d sha256={sha256} https://androzoo.uni.lu/api/download'
        down_command = down_command.format(sha256=sha256_str)
        os.system(down_command)

        try:
            path_apk = current_path + '/' + sha256_str + '.apk'
            print('==apk_decompile start==')
            path_apk_decompile = apk_decompile(path_apk, current_path)
            get_string_dex(path_apk_decompile)
            print('==apk_decompile finished==')
        except:
            pass


        try:
            print('==main_keywords_identity start==')
            target_keywords = main_keywords_identity(path_apk_decompile)
            print('==main_keywords_identity finished==')
        except:
            print('error: main_keywords_identity')

        cmd = 'echo finished {finished} >> finished.log'
        cmd = cmd.format(finished=finished)
        finished += 1
        os.system(cmd)

        print('finished all number: ', finished)

        rm_apk_command = 'rm ' + sha256_str + '.apk'
        rm_apk_decompile = 'rm -rf ' + sha256_str + '_compile'

        os.system(rm_apk_command)
        os.system(rm_apk_decompile)


def main(path_sha256_list, number_single_sha256, number_pool):
    """
    main function.

    :param path_sha256_list: <string>, path of list of pkl.
    :param number_single_sha256: <int>, The number of Apks processed by each core.
    :param number_pool: <int>, Number of cores. note: len(path_sha256_list) <= number_single_sha256*number_pool
    """
    starttime = datetime.datetime.now()
    cmd = 'echo start {start} >> time.txt'
    cmd = cmd.format(start=starttime)
    os.system(cmd)

    sha256_list = get_sha256(path_sha256_list)
    number_group = len(sha256_list) // number_single_sha256 + 1

    start = 0
    end = number_group
    data_list = []
    if start < len(sha256_list):
        for i in range(number_group):
            tmp = sha256_list[start:end]
            data_list.append(tmp)
            start = end
            end = end+number_single_sha256

    with Pool(number_pool) as p:
        p.map(main_single, data_list)

    endtime = datetime.datetime.now()
    diff_time = endtime - starttime
    cmd = 'echo end {end} >> time.txt'
    cmd = cmd.format(end=endtime)
    os.system(cmd)
    cmd = 'echo diff {diff} >> time.txt'
    cmd = cmd.format(diff=diff_time)
    os.system(cmd)


if __name__ == '__main__':
    main('sha256_list.pkl', 513, 50)
