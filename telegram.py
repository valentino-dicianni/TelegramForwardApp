import asyncio
import concurrent.futures
import io
import logging
import multiprocessing
import os

from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto

ERROR = -1
BASE_PATH = os.environ['APPDATA'] + "\\TelegramForward\\"
LOG_PATH = BASE_PATH + "Logs\\"
RULES_PATH = BASE_PATH + "rules.txt"
api_id = 1271225
api_hash = 'f36c296645a468c16a698ecb1e59e31b'
username = ""
user_id = ""


def string_to_bool(s_bool):
    if s_bool == 'True':
        return True
    else:
        return False


def get_username():
    return username


def get_user_id():
    return user_id


def get_confirmed_channels(uid):
    channel_list = {}
    open(f'{BASE_PATH}channel_list{uid}.txt', 'a', encoding="utf-8").close()

    with io.open(f'{BASE_PATH}channel_list{uid}.txt', 'r', encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("|")
            if len(fields) == 3 and string_to_bool(fields[2]):
                channel_list[fields[0]] = {"id": fields[0],
                                           "title": fields[1],
                                           "active": string_to_bool(fields[2])}
    return channel_list


def rows_from_channel_list(c_list):
    out = []
    for row in c_list:
        out_str = ""
        if c_list[row]['id'] != '':
            out_str += f"{c_list[row]['id']}"
        if c_list[row]['title'] != '':
            out_str += f" -- {c_list[row]['title']}"
        out.append((out_str, c_list[row]['active'], c_list[row]['id']))
    return out


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
    client = TelegramClient(BASE_PATH + "tc_session", api_id, api_hash, loop=loop)
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


async def start_telegram_loop(uid):
    """
    Event listener for messages received.
    Starts a new thread for the GUI process
    :return: void
    """
    logging.info(f"Telegram Client started Main Loop...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = TelegramClient(BASE_PATH + "tc_session", api_id, api_hash, loop=loop)
    await client.connect()
    me = await client.get_me()

    @client.on(events.NewMessage(func=lambda e: e.is_channel or e.is_group))
    async def handler(event):
        try:
            msg = event.original_update.message
            channel_name = ""
            sender = await event.get_sender()

            confirmed_channel_list = get_confirmed_channels(uid)

            logging.info(f"recived message from {sender.id}")
            if str(sender.id) in confirmed_channel_list:
                if msg.media is not None and isinstance(msg.media, MessageMediaPhoto):
                    print("send messsage with media photo")

                elif msg.reply_to is None:
                    print("forward only message")

        except Exception as e:
            logging.error("Error: " + str(e))

    @client.on(events.MessageEdited())
    async def handlerEdited(e):
        print("EDITED: " + str(e))

    await client.run_until_disconnected()


async def send_telegram_verification_code(cell_num):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = TelegramClient(BASE_PATH + "tc_session", api_id, api_hash, loop=loop)
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


async def check_telegram_user_state():
    """
    Check if the user is logged in
    :return: boolean check
    """
    global username, user_id
    logging.info("Telegram Client started User State...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = TelegramClient(BASE_PATH + "tc_session", api_id, api_hash, loop=loop)
    await client.connect()
    me = await client.get_me()
    if me is not None:
        username = str(me.username)
        user_id = str(me.id)

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
    client = TelegramClient(BASE_PATH + "tc_session", api_id, api_hash, loop=loop)
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
