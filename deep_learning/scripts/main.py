from datetime import datetime
import time
if __name__ == '__main__':
    start_time = datetime.now()
    
    time.sleep(3)    
    
    end = datetime.now()

    duration = end - start_time
    print(f"Duration: {duration}")    