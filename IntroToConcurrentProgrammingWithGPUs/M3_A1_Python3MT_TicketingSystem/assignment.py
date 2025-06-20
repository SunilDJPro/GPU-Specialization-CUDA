# Based on RealPython Threading Example page at https://realpython.com/intro-to-python-threading/ and
#   Python.org _thread library documentation at
#   https://docs.python.org/3/library/_thread.html?highlight=_thread#module-_thread
import logging
import random
import sys
import time
from threading import Thread, Lock

from core import Core

# Global shared variables for ticketing system
current_ticket = 0
ticket_lock = Lock()

def execute_ticketing_system_participation(ticket_number, part_id, shared_variable):
    output_file_name = "output-" + part_id + ".txt"
    # NOTE: Do not remove this print statement as it is used to grade assignment,
    # so it should be called by each thread
    print("Thread retrieved ticket number: {} started".format(ticket_number), file=open(output_file_name, "a"))
    time.sleep(random.randint(0, 5))
    
    # wait until your ticket number has been called
    global current_ticket, ticket_lock
    while True:
        with ticket_lock:
            if current_ticket == ticket_number:
                # output your ticket number and the current time
                break
        time.sleep(0.1)  # Small sleep to prevent busy waiting

    # NOTE: Do not remove this print statement as it is used to grade assignment,
    # so it should be called by each thread
    print("Thread with ticket number: {} completed".format(ticket_number), file=open(output_file_name, "a"))
    return 0


class Assignment(Core):

    USERNAME = "coder"
    active_threads = []

    def __init__(self, args):
        self.num_threads = 1
        self.args_conf_list = [['-n', 'num_threads', 1, 'number of concurrent threads to execute'],
                                ['-u', 'user', None, 'the user who is turning in the assignment, needs  to match the '
                                                    '.user file contents'],
                               ['-p', 'part_id', 'test', 'the id for the assignment, test by default']]
        super().__init__(self.args_conf_list)
        super().parse_args(args=args)
        _format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=_format, level=logging.INFO,
                            datefmt="%H:%M:%S")

    def run(self):
        output_file_name = "output-" + self.part_id + ".txt"
        open(output_file_name, 'w').close()
        if self.test_username_equality(self.USERNAME):
            sleeping_time = 0
            shared_variable = current_ticket  # Use the global current_ticket as shared variable
            
            for index in range(self.num_threads):
                logging.info("Assignment run    : create and start thread %d.", index)
                # This is where you will start a thread that will participate in a ticketing system
                # have the thread run the execute_ticketing_system_participation function
                ticket_number = index  # Assign ticket numbers starting from 0 (using index directly)
                thread = Thread(target=execute_ticketing_system_participation, 
                              args=(ticket_number, self.part_id, shared_variable))
                thread.start()
                self.active_threads.append(thread)

                # Threads will be given a ticket number and will wait until a shared variable is set to that number

                # The code will also need to know when all threads have completed their work
                sleeping_time += 1
            time.sleep(sleeping_time)
            self.manage_ticketing_system()
            logging.info("Assignment completed all running threads.")
            return 0
        else:
            logging.error("Assignment had an error your usernames not matching. Please check code and .user file.")
            return 1

    def manage_ticketing_system(self):
        # increment a ticket number shared by a number of threads and check that no active threads are running
        global current_ticket, ticket_lock
        
        for ticket_num in range(self.num_threads):  # Start from 0 to num_threads-1
            # Set the current ticket number to allow the corresponding thread to proceed
            with ticket_lock:
                current_ticket = ticket_num
            logging.info("Ticket number {} is now being served".format(ticket_num))
            time.sleep(1)  # Small delay between serving tickets
        
        # Wait for all threads to complete
        for thread in self.active_threads:
            thread.join()
        
        logging.info("All threads have completed their work")
        return 0


if __name__ == "__main__":
    assignment = Assignment(args=sys.argv[1:])
    exit_code = assignment.run()
    sys.exit(exit_code)