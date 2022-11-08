import os
import logging
from identification_config import Config
Cf = Config()


def run_w(shell_cmd):
    try:
        res = os.popen(shell_cmd).read().strip()
    except:
        logging.error("error in executing : " + shell_cmd)
        res = ""
    return res


def main_keywords_identity(path_apk_decompile):
    for keyword in Cf.end_model_format:
        shell_cmd = "ag %s -i -l --silent -m2 %s" % (keyword, path_apk_decompile)
        match = run_w(shell_cmd)
        if match:
            re = open(Cf.path_result, 'a')
            re.write('\n' + path_apk_decompile + '->' + 'keywords:' + keyword)
            re.close()
