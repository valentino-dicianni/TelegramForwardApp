import datetime
import re
import time
from logging.handlers import QueueListener, RotatingFileHandler
from multiprocessing import freeze_support

import PySimpleGUI as sg

from telegram import *

THEME = 'LightBlue'
ADD_WIN = None
rules_list = []
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U0001F1F2-\U0001F1F4"  # Macau flag
                           u"\U0001F1E6-\U0001F1FF"  # flags
                           u"\U0001F600-\U0001F64F"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U0001F1F2"
                           u"\U0001F1F4"
                           u"\U0001F620"
                           u"\u200d"
                           u"\u2640-\u2642"
                           "]+", flags=re.UNICODE)
special_char = re.compile("[^A-Za-z0-9ßćéāēèêáúüřťôíостанвиьпер. ,\n]+")

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


class Rule:
    def __init__(self, rule_id, form_id, to_ids=None, keywords=None):
        self.rule_id = rule_id
        self.from_id = form_id
        self.to_ids = [] if to_ids is None else to_ids
        self.keywords = [] if keywords is None else keywords

    @classmethod
    def from_string(cls, string_rule):
        rule_id = string_rule.split(')')[0]
        split_str = string_rule.split(')')[1].split('->')
        to_list = None
        keys_list = None
        if len(split_str) > 2:
            to_list = list(map(str.strip, split_str[2].split(",")))
        if len(split_str) > 3:
            keys_list = list(map(str.strip, split_str[3].split(",")))
        return cls(rule_id, split_str[1], to_list, keys_list)

    def rule_to_string(self):
        return f"{self.rule_id}) {self.from_id}->{','.join(self.to_ids)}->{','.join(self.keywords)}"


def logger_init():
    queue = multiprocessing.Queue()
    filename = datetime.datetime.now().strftime(f'{LOG_PATH}TClog_%d_%m_%Y.log')
    file_handler = RotatingFileHandler(filename, maxBytes=1048576, backupCount=5)
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(queue, file_handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return ql, queue


q_listener, q = logger_init()


def list_to_string(l, sep):
    return sep.join(l)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def dump_rules_to_file(rules):
    try:
        logging.info("Dumping rules to file...")
        with io.open(RULES_PATH, 'w+', encoding="utf-8") as file:
            for r in rules:
                file.write(r.rule_to_string())
                file.write("\n")
        return True
    except Exception as e:
        logging.exception(e)
        return False


def get_rules_from_file():
    rules_result = []

    if not os.path.exists(RULES_PATH):
        return []

    try:
        with io.open(RULES_PATH, 'r', encoding="utf-8") as file:
            for line in file:
                r = line.strip()
                rules_result.append(Rule.from_string(r))
        return rules_result
    except Exception as e:
        logging.exception(e)
        return []


def get_rows_from_channel_list(c_list):
    out = []
    for row in c_list:
        out_str = ""
        if c_list[row]['id'] != '':
            out_str += f"{c_list[row]['id']}"
        if c_list[row]['title'] != '':
            out_str += f" -- {c_list[row]['title']}"
        out.append((out_str, c_list[row]['active'], c_list[row]['id']))
    return out


def create_channel_item(text, checked, box_id):
    return [sg.CBox(f'{text}', default=checked, key=box_id)]


def get_add_rule_layout():
    chats = rows_from_channel_list(get_all_channels())

    chats_desc = list(map(lambda x: x[0], chats))

    layout = [
        [sg.Text('New Rule', justification="center", text_color="darkblue", size=(None, 2), font=("", 22, "bold"))],

        [sg.Text("Select the source for your rule:")],
        [sg.Combo(chats_desc, key="source")],

        [sg.Text("Select the destination for your rule:")],
        [sg.Combo(chats_desc, key="destination")],

        [sg.Text("Add some filtering keywords separated with comma:")],
        [sg.InputText(key="keywords")],
        [sg.HorizontalSeparator()],
        [sg.Button('Save Rule', key="save_rule", size=(12, 1)),
         sg.Button('Close', button_color=("white", "red"), key="close", size=(12, 1))]
    ]
    return layout


def get_main_layout():
    global rules_list
    rules_string = list(map(lambda x: x.rule_to_string(), rules_list))
    header = [
        [sg.Text('Forward Rules', justification="center", text_color="darkblue", size=(None, 2),
                 font=("", 22, "bold"))],
    ]

    rules_layout = [
        [
            sg.Listbox(rules_string, select_mode='extended', key='rules_list', background_color="white", size=(60, 15))
        ]
    ]

    buttons = [
        [
            sg.Button('Add Rule', key="add_rule", size=(12, 1), ),
            sg.Button('Remove Rule', key="remove_rule", size=(12, 1)),
            sg.Button('Log Out', key="logout", size=(12, 1), button_color=('white', 'red'), pad=((130, 0), (0, 0)))
        ],
    ]

    left_side = [
        [sg.Image('utils_files\\logoFS.png', key='-PICTUREFRAME-', size=(150, 100), pad=((10, 10), (160, 0)))],
        [sg.Text("Connected as " + get_username(), text_color="green", size=(25, 1), pad=((0, 0), (100, 0)))],
    ]

    right_side = [
        [sg.Column(header)],
        [sg.Column(rules_layout)],
        [sg.Column(buttons, pad=((0, 0), (10, 0)))],
    ]

    layout = [
        [
            sg.Column(left_side, key="image"),
            sg.VerticalSeparator(),
            sg.Column(right_side, visible=True, key='buttons'),
        ],
    ]

    return layout


def get_auth_layout():
    file_list_column = [
        [sg.Image('utils_files\\logoFS.png', key='-PICTUREFRAME2-', size=(150, 100), pad=(150, 40))],
        [
            sg.Text('Phone Number:', key="phoneText", size=(20, 1), pad=((40, 0), (0, 0))),
            sg.InputText("+39", key='inputNumber', size=(30, 1)),
        ],
        [
            sg.Text('Verification Code:', key="codeText", size=(20, 1), pad=((40, 0), (0, 0)), visible=True),
            sg.InputText(key='codeInput', size=(30, 1), disabled=True),
        ],

        [
            sg.Button('Get Code', key='codeBtn', size=(8, 1), pad=((150, 0), (50, 0))),
            sg.Button('Log In', key='loginBtn', size=(8, 1), pad=((8, 0), (50, 0)), disabled=True),

        ],
        [
            sg.Button('Exit', size=(8, 1), pad=((190, 0), (20, 0))),
            sg.Text(text="ERROR: wrong number", key='ERROR', size=(20, 1), visible=False, text_color='Red')
        ]
    ]

    layout = [
        [

            sg.Column(file_list_column),
        ],
    ]
    return layout


def bind_main_UI(telegram_process):
    global rules_list, ADD_WIN

    logout_flag = False
    running_process = telegram_process
    rules_list = get_rules_from_file()
    sg.theme(THEME)
    main_win = sg.Window('Telegram Forward',
                         get_main_layout(),
                         icon='utils_files\\logoFS.ico',
                         finalize=True
                         )
    logging.info("Binding Main UI...")
    try:
        while True:
            window, event, values = sg.read_all_windows()

            # Main Window
            if window == main_win:

                if event in (sg.WIN_CLOSED, 'Exit'):
                    dump_rules_to_file(rules_list)
                    running_process.terminate()

                    while running_process.is_alive():
                        time.sleep(0.1)
                        logging.info("Killing telegram Process...")
                    break

                if event in 'logout':
                    logging.info("Logout button pressed...")
                    running_process.terminate()
                    while running_process.is_alive():
                        time.sleep(0.1)
                        logging.info("Killing telegram Process...")

                    logout_flag = True
                    break

                if event == "add_rule":
                    ADD_WIN = sg.Window('Telegram Forward',
                                        get_add_rule_layout(),
                                        icon='utils_files\\logoFS.ico',
                                        finalize=True
                                        )
                if event == "remove_rule":
                    for elem in values['rules_list']:
                        rules_list.pop(int(elem.split(')')[0]) - 1)
                        logging.info("Removed rule: " + elem)

                    dump_rules_to_file(rules_list)
                    main_win['rules_list'].Update(list(map(lambda x: x.rule_to_string(), rules_list)))

            # Add rule window
            if window == ADD_WIN:
                if event in (sg.WIN_CLOSED, 'close'):
                    window.close()

                if event == "save_rule":
                    key_list = values['keywords'].strip().split(',')
                    dest_list = [values['destination'].strip().split()[0]]
                    new_r = Rule(len(rules_list) + 1, values['source'].split()[0], dest_list, key_list)
                    rules_list.append(new_r)
                    logging.info("Added new rule: " + new_r.rule_to_string())
                    dump_rules_to_file(rules_list)
                    main_win['rules_list'].Update(list(map(lambda x: x.rule_to_string(), rules_list)))
                    window.close()

        if logout_flag:
            os.remove(BASE_PATH + "tc_session.session")
            if os.path.exists(BASE_PATH + "tc_session.session-journal"):
                os.remove(BASE_PATH + "tc_session.session-journal")
            window.close()
            logging.info("Session files deleted, starting bindUI...")
            bind_auth_UI()

    except Exception as e:
        logging.exception(e)
    main_win.close()


def bind_auth_UI():
    phone_number = ""
    hash_code = ""
    start = False
    try:  # Client already Logged IN
        if check_status_start_thread():
            # sg.theme_previewer()
            logging.warning("Already logged in...")
            proc = telegram_loop_start_process()
            bind_main_UI(proc)

        else:  # New Client
            sg.theme(THEME)
            # sg.theme_previewer()
            logging.warning("Not logged in...")
            window = sg.Window('Telegram Copier - Authentication', get_auth_layout(), icon='utils_files\\logoFS.ico',
                               finalize=True, size=(500, 400))
            while True:
                event, values = window.read(timeout=100)
                if event in (sg.WIN_CLOSED, 'Exit'):
                    window.close()
                    break

                if event in 'codeBtn':
                    phone_number = values['inputNumber']
                    if validate_number(phone_number):
                        response = t_send_code_start_thread(phone_number)
                        if response != ERROR:
                            logging.info(response)
                            hash_code = response
                            window.Element('codeInput').Update(disabled=False)
                            window.Element('loginBtn').Update(disabled=False)
                        else:
                            sg.Popup('Error!', 'Wrong number.')

                if event in 'loginBtn':
                    if not start:
                        verification_code = values['codeInput']
                        response = t_sign_in_start_thread(phone_number, int(verification_code), hash_code)
                        if response != ERROR:
                            start = True
                            break
                        else:
                            sg.Popup('Error!', 'Wrong validation code.')
                    else:
                        sg.Popup('Error!', 'Reload app.')

            if start:
                res = get_channels_start_thread()
                if res != ERROR:
                    proc = telegram_loop_start_process()
                    window.close()
                    bind_main_UI(proc)
                else:
                    sg.Popup('Error!', 'Fetching Channels. Please Exit and retry.')
                    window.close()
                    logging.error("ERROR FETCHING CHANNELS")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    freeze_support()
    logging.info("======### SESSION START ###=====")
    logging.info("Building UI...")
    bind_auth_UI()
    logging.info("======### SESSION END ###=====\n")
    q_listener.stop()
