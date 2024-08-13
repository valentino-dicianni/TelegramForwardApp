import asyncio
import concurrent.futures
import datetime
import io
import logging
import multiprocessing
import os
import sys
import time
from logging.handlers import QueueListener, RotatingFileHandler
from multiprocessing import freeze_support

import PySimpleGUI as sg
from PIL import Image, ImageDraw
from openai import OpenAI
from telethon import TelegramClient, events

# THEME = 'SystemDefault1'
THEME = 'LightGray3'

ADD_WIN = None
rules_list = []

ERROR = -1
BASE_PATH = os.environ['APPDATA'] + "\\TelegramForward\\"
LOG_PATH = BASE_PATH + "Logs\\"
RULES_PATH = BASE_PATH + "rules.txt"

api_id = 1271225
api_hash = 'f36c296645a468c16a698ecb1e59e31b'
username = ""
user_id = ""
profile_photo_path = BASE_PATH + "profile_photo.png"

client = OpenAI(
    api_key="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",
)

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


class Rule:
    def __init__(self, rule_id, form_id, to_id, prompt_gpt="", keywords=None):
        self.rule_id = rule_id
        self.from_id = int(form_id)
        self.to_id = int(to_id)
        self.keywords = [] if keywords is None else keywords
        self.prompt_gpt = prompt_gpt

    @classmethod
    def from_string(cls, index, string_rule):
        split_str = string_rule.split('->')
        from_id = split_str[0]
        to_id = split_str[1]
        keys_list = None
        prompt = None
        if len(split_str) > 2 and split_str[2].strip() != "":
            keys_list = list(map(lambda x: x.strip(), split_str[2].split(",")))
        if len(split_str) > 3 and split_str[3].strip() != "":
            prompt = split_str[3].strip()
        return cls(index, from_id, to_id, prompt_gpt=prompt, keywords=keys_list)

    def rule_to_string(self):
        base_str = f"From: {self.from_id} To: {self.to_id}"
        if len(self.keywords) > 0:
            base_str += f" - Filter: {','.join(self.keywords)}"
        if self.prompt_gpt != "":
            base_str += f" - PromptGPT: {self.prompt_gpt}"

        return base_str

    def rule_to_file(self):
        return f"{self.from_id}->{self.to_id}->{','.join(self.keywords)}->{self.prompt_gpt}"


def logger_init():
    queue = multiprocessing.Queue()
    filename = datetime.datetime.now().strftime(f'{LOG_PATH}TFlog_%d_%m_%Y.log')
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


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def list_to_string(l, sep):
    return sep.join(l)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def string_to_bool(s_bool):
    if s_bool == 'True':
        return True
    else:
        return False


def get_username():
    return username


def get_user_id():
    return user_id


def get_all_channels():
    global user_id
    c_list = {}
    io.open(f'{BASE_PATH}channel_list{user_id}.txt', 'a', encoding="utf-8").close()
    with io.open(f'{BASE_PATH}channel_list{user_id}.txt', 'r', encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("|")
            if len(fields) == 3:
                c_list[fields[0]] = {
                    "id": fields[0],
                    "title": fields[1],
                    "active": string_to_bool(fields[2])
                }
    return c_list


def persist_channels(dialogs):
    global user_id
    file_content = ""
    io.open(f'{BASE_PATH}channel_list{user_id}.txt', 'a').close()
    with io.open(f'{BASE_PATH}channel_list{user_id}.txt', "r", encoding="utf-8") as file:
        for dialog in dialogs:
            c_id = dialog.entity.id
            c_t = dialog.entity.title
            found = False
            file.seek(0, 0)
            for line in file:
                line_list = line.strip().split('|')
                if str(c_id) in line_list[0]:
                    file_content += f"{c_id}|{line_list[1]}|{line_list[2]}\n"
                    found = True
                    break
            if not found:
                file_content += f"{c_id}|{c_t}|{False}\n"

    if file_content != "":
        with io.open(f'{BASE_PATH}channel_list{user_id}.txt', 'w', encoding="utf-8") as file:
            file.write(file_content)
    return True


def sort_channels_file():
    global user_id
    file_content = ""
    io.open(f'{BASE_PATH}channel_list{user_id}.txt', 'a').close()
    with io.open(f'{BASE_PATH}channel_list{user_id}.txt', "r", encoding="utf-8") as file:
        for line in file:
            line_list = line.strip().split('|')
            if line_list[2] == "True":
                file_content = f"{line_list[0]}|{line_list[1]}|{line_list[2]}\n" + file_content
            else:
                file_content = file_content + f"{line_list[0]}|{line_list[1]}|{line_list[2]}\n"

    if file_content != "":
        with io.open(f'{BASE_PATH}channel_list{user_id}.txt', 'w', encoding="utf-8") as file:
            file.write(file_content)
    return True


async def get_channels_from_telegram():
    channels = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = TelegramClient(BASE_PATH + "tf_session", api_id, api_hash, loop=loop)
    try:
        logging.info("Telegram Client started Getting Channels...")

        await client.connect()
        async for dialog in client.iter_dialogs():
            if not dialog.is_group and dialog.is_channel:
                channels.append(dialog)
        persist_channels(channels)
        sort_channels_file()
        await client.disconnect()
        return len(channels)
    except Exception as e:
        await client.disconnect()
        logging.exception(e)


def find_rule_by_sender_id(uid, r_list):
    match_rules = []
    for r in r_list:
        if int(r.from_id) == int(uid):
            match_rules.append(r)

    if len(match_rules) > 0:
        return True, match_rules
    else:
        return False, None


def find_keyword_in_msg(rule, text):
    if len(rule.keywords) == 0:
        return True

    for k in rule.keywords:
        if k.lower() in text.lower():
            return True

    return False


def gpt4_query(prompt, max_tokens=100):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


async def start_telegram_loop(uid):
    """
    Event listener for messages received.
    Starts a new thread for the GUI process
    :return: void
    """
    logging.info(f"Telegram Client started Main Loop...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = TelegramClient(BASE_PATH + "tf_session", api_id, api_hash, loop=loop)
    await client.connect()
    me = await client.get_me()

    @client.on(events.NewMessage(func=lambda e: e.is_channel or e.is_group))
    async def handler(event):
        try:
            sender = await event.get_sender()

            rules_local = get_rules_from_file()
            logging.info(f"recived message from {sender.id}")

            is_rule, rules = find_rule_by_sender_id(sender.id, rules_local)

            if is_rule:
                msg = event.message
                for r in rules:
                    to_chat = int(r.to_id)

                    if find_keyword_in_msg(r, msg.text):
                        if r.prompt_gpt != "":
                            final_prompt = f"{str(r.prompt_gpt)} '{str(msg.text)}'"
                            logging.info(final_prompt)
                            gpt_res = gpt4_query(final_prompt)

                            await client.send_message(to_chat, gpt_res, silent=False)
                        else:
                            await client.send_message(to_chat, msg, silent=False)

        except Exception as e:
            logging.error("Error: " + str(e))

    @client.on(events.MessageEdited())
    async def handlerEdited(e):
        print("EDITED: " + str(e))

    await client.run_until_disconnected()


async def send_telegram_verification_code(cell_num):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = TelegramClient(BASE_PATH + "tf_session", api_id, api_hash, loop=loop)
    try:
        logging.info("Telegram Client started Verification Code...")
        await client.connect()

        if not await client.is_user_authorized():
            hash_code = await client.sign_in(cell_num)
            logging.warning("In thread: " + str(hash_code.phone_code_hash))
            await client.disconnect()
            return hash_code.phone_code_hash
        else:
            await client.disconnect()
            return ERROR
    except Exception as e:
        await client.disconnect()
        logging.error(e)
        return ERROR


def create_rounded_image(input_image_path, output_image_path, final_size=(40, 40)):
    with Image.open(input_image_path) as img:
        larger_size = (final_size[0] * 2, final_size[1] * 2)
        img = img.resize(larger_size, Image.Resampling.LANCZOS)

        # Crea una maschera circolare
        mask = Image.new('L', larger_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, larger_size[0], larger_size[1]), fill=255)

        # Applica la maschera all'immagine
        circular_image = Image.new('RGBA', larger_size)
        circular_image.paste(img, (0, 0), mask)

        # Ridimensiona l'immagine circolare alla dimensione finale
        circular_image = circular_image.resize(final_size, Image.Resampling.LANCZOS)

        # Salva l'immagine risultante
        circular_image.save(output_image_path, 'PNG', quality=100)


async def check_telegram_user_state():
    """
    Check if the user is logged in
    :return: boolean check
    """
    global username, user_id
    logging.info("Telegram Client started User State...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = TelegramClient(BASE_PATH + "tf_session", api_id, api_hash, loop=loop)
    await client.connect()
    me = await client.get_me()

    if me is not None:
        username = str(me.username)
        user_id = str(me.id)
        user_photo = await client.download_profile_photo('me')
        create_rounded_image(user_photo, profile_photo_path)
        os.remove(user_photo)

    if await client.is_user_authorized():
        await client.disconnect()
        return True
    else:
        await client.disconnect()
        return False


async def telegram_sign_in(phone_number, verification_code, hash_code):
    """
    Send Sign In request.
    :return: User object containing all information about user
    """
    logging.info("Telegram Client started Sign In...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    global username, user_id
    client = TelegramClient(BASE_PATH + "tf_session", api_id, api_hash, loop=loop)
    try:
        await client.connect()
        res = await client.sign_in(phone_number, verification_code, phone_code_hash=hash_code)
        logging.info(f"Sign In as {res}")
        username = str(res.username)
        user_id = str(res.id)
        await client.disconnect()
        return 1
    except Exception as e:
        await client.disconnect()
        logging.error(e)
        return ERROR


def telegram_loop_process(uid):
    try:
        asyncio.run(start_telegram_loop(uid))
    except Exception as e:
        logging.exception(e)


def telegram_loop_start_process():
    """
        t = threading.Thread(target=telegram_loop_thread, args=(), kwargs={})
        t.setDaemon(True)
        t.start()
    """
    # global telegram_process
    proc = multiprocessing.Process(target=telegram_loop_process, args=(user_id,))
    proc.daemon = True
    proc.start()
    logging.info("Starting Telegram process")
    return proc


def t_send_code_start_thread(num):
    def send_code_thread(cell):
        try:
            result = asyncio.run(send_telegram_verification_code(cell))
            return result
        except Exception as e:
            logging.error(e)
            return ERROR

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(send_code_thread, num)
        return future.result()


def t_sign_in_start_thread(number, verification, hash_c):
    def sign_in_thread(n, v, h):
        try:
            result = asyncio.run(telegram_sign_in(n, v, h))
            return result
        except Exception as e:
            logging.error(e)
            return ERROR

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(sign_in_thread, number, verification, hash_c)
        return future.result()


def check_status_start_thread():
    def is_logged_in_thread():
        try:
            result = asyncio.run(check_telegram_user_state())
            return result
        except Exception as e:
            logging.error(e)
            return ERROR

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(is_logged_in_thread)
        return future.result()


def get_channels_start_thread():
    def get_channels_thread():
        try:
            result = asyncio.run(get_channels_from_telegram())
            return result
        except Exception as e:
            logging.exception(e)
            return ERROR

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_channels_thread)
        return future.result()


def validate_number(phone_number):
    if phone_number.startswith('+') and phone_number[1:].isnumeric() and len(phone_number) >= 10:
        return True
    else:
        return False


def validate_code(code):
    if code.isnumeric() and len(code) == 5:
        return True
    else:
        return False


def dump_rules_to_file(rules, path=None):
    if path is None:
        path = RULES_PATH
    try:
        logging.info("Dumping rules to file...")
        with io.open(path, 'w+', encoding="utf-8") as file:
            for r in rules:
                file.write(r.rule_to_file())
                file.write("\n")
        return True
    except Exception as e:
        logging.exception(e)
        return False


def get_rules_from_file(path=None):
    rules_result = []
    if path is None:
        path = RULES_PATH

    if not os.path.exists(path):
        return []

    try:
        with io.open(path, 'r', encoding="utf-8") as file:
            for i, line in enumerate(file):
                r = line.strip()
                rules_result.append(Rule.from_string(i, r))
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
            out_str += f" - {c_list[row]['title']}"
        out.append(out_str)
    return out


def get_selected_row_from_channel_list(c_list, c_id):
    for row in c_list:
        if int(c_list[row]['id']) == c_id:
            return f"{c_list[row]['id']} - {c_list[row]['title']}"
    return ""


def item_row(item_num, chats):
    layout = [
        [
            sg.B('❌', button_color=(sg.theme_text_color(), sg.theme_background_color()), enable_events=True,
                 k=('-DEL-', item_num), tooltip='Delete Rule'),
            sg.B('✏️', button_color=(sg.theme_text_color(), sg.theme_background_color()), enable_events=True,
                 k=('edit', item_num), tooltip='Edit Rule'),
            sg.B('✔️', button_color=(sg.theme_text_color(), sg.theme_background_color()), enable_events=True,
                 k=('save', item_num), tooltip='Save Changes')
        ],
        [
            sg.Text(f'Source: ', k=('source', item_num), size=(15, 1)),
            sg.Combo(chats, size=(35, 1), k=('source_in', item_num)),
        ],
        [
            sg.Text(f'Destination: ', k=('destination', item_num), size=(15, 1)),
            sg.Combo(chats, size=(35, 1), k=('destination_in', item_num)),
        ],
        [
            sg.Text(f'Filter Words: ', k=('filter', item_num), size=(15, 1)),
            sg.Input(size=(35, 1), k=('filter_in', item_num)),
        ],
        [
            sg.Text(f'GPT Modification: ', k=('gpt', item_num), size=(15, 1)),
            sg.Multiline(size=(35, 3), k=('gpt_in', item_num)),
        ]
    ]

    framed_layout = [[sg.Frame("", layout)]]
    row = [sg.pin(sg.Col(framed_layout, k=('-ROW-', item_num)))]
    return row


def change_mod_rule(w, num_rule, disable=False):
    w[('source_in', num_rule)].update(disabled=disable)
    w[('destination_in', num_rule)].update(disabled=disable)
    w[('filter_in', num_rule)].update(disabled=disable)
    w[('gpt_in', num_rule)].update(disabled=disable)


# Frame Based dashboard

def get_main_layout():
    header = [
        [sg.Text('Forward Rules', justification="center", text_color="darkblue", font=("", 22, "bold"))],
    ]

    left_side = [
        [sg.Image(source=resource_path('logoFS.png'), key='-PICTUREFRAME-', size=(150, 150), pad=(0, 30))],

        [sg.Button('➕   New Rule', key=("add_rule", 0), size=(20, 1), )],
        [sg.Button('⬇️   Download', key=("download_rules", 0), size=(20, 1), )],
        [sg.Button('⬆️     Upload  ', key=("upload_rules", 0), size=(20, 1))],
        [sg.Button('↪️    Log Out ', key=("logout", 0), size=(20, 1), button_color=('white', 'maroon'))],

        [
            sg.Image(source=profile_photo_path, key='-PICTUREFRAME-', pad=((0, 0), (40, 10))),
            sg.Text(f"Connected as:\n{get_username()}", text_color="green", pad=((10, 0), (40, 10)))
        ],
    ]

    right_side = [
        [sg.Column(header)],
        [sg.Column([],
                   scrollable=True,
                   vertical_scroll_only=True,
                   k='-RULE SECTION-',
                   size=(450, 380),
                   expand_y=True,
                   pad=(10, 0),
                   vertical_alignment='top'
                   )],
    ]

    layout = [
        [
            sg.Column(left_side, key="image", vertical_alignment='bottom', element_justification='center'),
            sg.VerticalSeparator(),
            sg.Column(right_side, visible=True, key='buttons', vertical_alignment='bottom'),
        ],
    ]

    return layout


def get_auth_layout():
    file_list_column = [
        [sg.Image(resource_path('logoFS.png'), key='-PICTUREFRAME2-', size=(150, 150), pad=((0, 0), (0, 30)))],
        [
            sg.Text('Phone Number:', key="phoneText", size=(20, 1)),
            sg.InputText("+39", key='inputNumber', size=(20, 1)),
        ],
        [
            sg.Text('Verification Code:', key="codeText", size=(20, 1), visible=True),
            sg.InputText(key='codeInput', size=(20, 1), disabled=True),
        ],

        [
            sg.Button('Get Code', key='codeBtn', size=(8, 1), pad=(5, (20, 5))),
            sg.Button('Log In', key='loginBtn', size=(8, 1), pad=(5, (20, 5)), disabled=True),
        ],
        [
            sg.Button('Exit', size=(8, 1)),
            sg.Text(text="ERROR: wrong number", key='ERROR', size=(20, 1), visible=False, text_color='Red')
        ]
    ]

    layout = [
        [
            sg.Column(file_list_column, vertical_alignment='top', element_justification='center'),
        ],
    ]
    return layout


def upsert_rule(new_r: Rule):
    for r in rules_list:
        if r.rule_id == new_r.rule_id:
            r.from_id = new_r.from_id
            r.to_id = new_r.to_id
            r.keywords = new_r.keywords
            r.prompt_gpt = new_r.prompt_gpt
            return
    # if not found
    rules_list.append(new_r)


def remove_rule(r_id):
    for r in rules_list:
        if r.rule_id == r_id:
            rules_list.remove(r)
            break


def bind_main_UI(telegram_process):
    global rules_list, ADD_WIN

    logout_flag = False
    running_process = telegram_process
    rules_list = get_rules_from_file()
    sg.theme(THEME)
    main_win = sg.Window('Telegram Forward',
                         get_main_layout(),
                         icon=resource_path('logoFS.ico'),
                         finalize=True,
                         metadata=0,
                         )
    for r in rules_list:
        main_win.metadata = max(main_win.metadata, r.rule_id)

        channels = get_rows_from_channel_list(get_all_channels())
        main_win.extend_layout(main_win['-RULE SECTION-'], [item_row(r.rule_id, channels)])
        main_win[('source_in', r.rule_id)].update(get_selected_row_from_channel_list(get_all_channels(), r.from_id))
        main_win[('destination_in', r.rule_id)].update(get_selected_row_from_channel_list(get_all_channels(), r.to_id))
        main_win[('filter_in', r.rule_id)].update(",".join(r.keywords))
        main_win[('gpt_in', r.rule_id)].update(r.prompt_gpt)
        change_mod_rule(main_win, r.rule_id, disable=True)

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

                if event[0] == 'logout':
                    logging.info("Logout button pressed...")
                    running_process.terminate()
                    while running_process.is_alive():
                        time.sleep(0.1)
                        logging.info("Killing telegram Process...")

                    logout_flag = True
                    break

                if event[0] == "add_rule":
                    window.metadata += 1
                    channels = get_rows_from_channel_list(get_all_channels())
                    window.extend_layout(window['-RULE SECTION-'], [item_row(window.metadata, channels)])
                    window.visibility_changed()
                    window['-RULE SECTION-'].contents_changed()

                elif event[0] == '-DEL-':
                    window[('-ROW-', event[1])].update(visible=False)
                    remove_rule(event[1])
                    dump_rules_to_file(rules_list)
                    window['-RULE SECTION-'].contents_changed()

                elif event[0] == 'edit':
                    change_mod_rule(window, event[1], disable=False)

                elif event[0] == 'save':
                    if values[('source_in', event[1])] == "" or values[('destination_in', event[1])] == "":
                        sg.popup("Warning: source and destination should not be  empty!",
                                 icon=resource_path('logoFS.ico')
                                 )
                    else:
                        change_mod_rule(window, event[1], disable=True)
                        keywords = [] if values[('filter_in', event[1])].strip() == "" else values[
                            ('filter_in', event[1])].strip().split(',')
                        source_channel = values[('source_in', event[1])].split()[0]
                        dest_channel = values[('destination_in', event[1])].split()[0]
                        prompt = values[('gpt_in', event[1])]

                        new_r = Rule(event[1], source_channel, dest_channel, keywords=keywords, prompt_gpt=prompt)
                        upsert_rule(new_r)
                        logging.info("Added new rule: " + new_r.rule_to_string())
                        dump_rules_to_file(rules_list)

                if event[0] == "download_rules":
                    file_chosen = sg.popup_get_file('Save as: ', save_as=True, no_window=True,
                                                    icon=resource_path('logoFS.ico'))
                    if file_chosen:
                        dump_rules_to_file(rules_list, file_chosen + ".txt")

                if event[0] == "upload_rules":
                    file_chosen = sg.popup_get_file('', no_window=True,
                                                    icon=resource_path('logoFS.ico'))
                    if file_chosen:
                        rules_list = get_rules_from_file(file_chosen)

        if logout_flag:
            os.remove(BASE_PATH + "tf_session.session")
            if os.path.exists(BASE_PATH + "tf_session.session-journal"):
                os.remove(BASE_PATH + "tf_session.session-journal")
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
            logging.warning("Already logged in...")

            # Protezione
            if username != "valentino_dicianni" and username != "Michjr17" and username != "ChrBosco":
                sg.Popup('Error!', 'Account not allowed.',
                         icon=resource_path('logoFS.ico'))
                os.remove(BASE_PATH + "tf_session.session")
                if os.path.exists(BASE_PATH + "tf_session.session-journal"):
                    os.remove(BASE_PATH + "tf_session.session-journal")
                return False

            res = get_channels_start_thread()
            if res != ERROR:
                proc = telegram_loop_start_process()
                bind_main_UI(proc)

        else:  # New Client
            sg.theme(THEME)
            # sg.theme_previewer()
            logging.warning("Not logged in...")
            window = sg.Window('Telegram Copier - Authentication', get_auth_layout(),
                               icon=resource_path('logoFS.ico'),
                               finalize=True)
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
                            sg.Popup('Error!', 'Wrong number.',
                                     icon=resource_path('logoFS.ico'))

                if event in 'loginBtn':
                    if not start:
                        verification_code = values['codeInput']
                        response = t_sign_in_start_thread(phone_number, int(verification_code), hash_code)
                        if response != ERROR:
                            start = True
                            break
                        else:
                            sg.Popup('Error!', 'Wrong validation code.',
                                     icon=resource_path('logoFS.ico'))
                    else:
                        sg.Popup('Error!', 'Reload app.',
                                 icon=resource_path('logoFS.ico'))

            if start:
                res = get_channels_start_thread()
                if res != ERROR:
                    proc = telegram_loop_start_process()
                    window.close()
                    bind_main_UI(proc)
                else:
                    sg.Popup('Error!', 'Fetching Channels. Please Exit and retry.',
                             icon=resource_path('logoFS.ico'))
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
