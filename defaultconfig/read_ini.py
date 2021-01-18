
import configparser
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

def read_properties_from_config_ini(config_name):
    config = configparser.ConfigParser()
    # -read读取ini文件
    config.read('defaultconfig/config_{}.ini'.format(config_name), encoding='utf-8')
    config_tup = {}
    print("====================================")
    print("config file properties:")
    print("====================================")
    for sections in config.sections():
        for items in config.items(sections):
            print(items)
            if not config_tup.__contains__(sections):
                config_tup[sections] = {}
            config_tup[sections][items[0]] = items[1]


    return config_tup
