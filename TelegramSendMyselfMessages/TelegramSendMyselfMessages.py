from telegram import Bot

class TelegramSendMyselfMessages:

    def __init__(self):
        self.bot = Bot(token='')


    def sendMessageToMe(self, message):
        self.bot.send_message(chat_id='', text=str(message))
