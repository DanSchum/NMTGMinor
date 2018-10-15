from telegram import Bot

class TelegramSendMyselfMessages:

    def __init__(self):
        self.bot = Bot(token='672011097:AAGwcWOu8GXpOx4t77M_-dtuk9QqCu3CHQw')


    def sendMessageToMe(self, message):
        self.bot.send_message(chat_id='630191534', text=str(message))
