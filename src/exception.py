# contains Python runtime events ( Exception, etc)
import sys
from logger import logging

def error_message_detail( error, error_detail:sys):
    
    # info in the occurred error
    _,_,exc_tb = error_detail.exc_info()
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    message = "\n Error occurred , filename = {0}, at line NO. {1}, \n error= {2}".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return message
    

# code for Custom Exception to be usd for Logger
class CustomException(Exception):
    def __init__(self, error, error_details:sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_details)
        
    
    def __str__(self):
        return self.error_message



if __name__ == "__main__":    
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)