# Based on code from https://github.com/Nohclu/Sleeping-Barber-Python-3.6-/blob/master/barber.py
import time
import random
import threading
from queue import Queue

CUSTOMERS_SEATS = 15        # Number of seats in BarberShop
BARBERS = 3                # Number of Barbers working
EVENT = threading.Event()   # Event flag, keeps track of Barber/Customer interactions
Earnings = 0
SHOP_OPEN = True
earnings_lock = threading.Lock()  # Lock for thread-safe earnings updates


class Customer(threading.Thread):       # Producer Thread
    def __init__(self, queue):          # Constructor passes Global Queue (all_customers) to Class
        threading.Thread.__init__(self)
        self.queue = queue
        self.rate = self.what_customer()

    @staticmethod
    def what_customer():
        customer_types = ["adult", "senior", "student", "child"]
        customer_rates = {"adult": 16,
                          "senior": 7,
                          "student": 10,
                          "child": 7}
        t = random.choice(customer_types)
        print(f"{t} customer arrives - rate: €{customer_rates[t]}")
        return customer_rates[t]

    def run(self):
        if not self.queue.full():  # Check queue size
            self.queue.put(self)    # Add this customer to the queue
            print(f"Customer added to queue. Queue size: {self.queue.qsize()}")
            EVENT.set()  # Signal that a customer is available
        else:
            # If Queue is full, Customer leaves.
            print("Queue full, customer has left.")

    def trim(self):
        print("Customer haircut started.")
        haircut_time = 3 * random.random()  # Random haircut duration
        time.sleep(haircut_time)  # Simulate time for haircut
        
        print(f"Haircut finished. Haircut took {haircut_time:.2f} seconds")
        
        # Thread-safe earnings update
        global Earnings
        with earnings_lock:
            Earnings += self.rate
            print(f"Payment received: €{self.rate}. Total earnings: €{Earnings}")


class Barber(threading.Thread):     # Consumer Thread
    def __init__(self, queue, barber_id):      # Constructor passes Global Queue (all_customers) to Class
        threading.Thread.__init__(self)
        self.queue = queue  # Set queue property
        self.barber_id = barber_id
        self.name = f"Barber-{barber_id}"
        self.sleep = True   # No Customers in Queue therefore Barber sleeps by default

    def is_empty(self):  # Simple function that checks if there is a customer in the Queue
        if self.queue.empty():
            self.sleep = True   # If nobody in the Queue Barber sleeps.
        else:
            self.sleep = False  # Else he wakes up.
        print(f"{self.name} sleep status: {self.sleep}")

    def run(self):
        global SHOP_OPEN
        print(f"{self.name} starts working")
        
        while SHOP_OPEN:
            # Check if queue is empty
            if self.queue.empty():
                print(f"{self.name} is sleeping... waiting for customers")
                EVENT.wait()  # Wait for customer signal
                EVENT.clear()  # Reset the event for next customer
            
            # Check again after waking up (shop might be closed)
            if not SHOP_OPEN:
                break
                
            # Try to get a customer from queue
            if not self.queue.empty():
                print(f"{self.name} is awake and serving customer")
                customer = self.queue.get()  # Get customer from queue
                self.is_empty()
                customer.trim()  # Customer's hair is being cut
                self.queue.task_done()  # Mark task as done
                print(f"Customer served by {self.name}")
            
        print(f"{self.name} finished working")


def wait():
    time.sleep(1 * random.random())


if __name__ == '__main__':
    Earnings = 0
    SHOP_OPEN = True
    barbers = []
    all_customers = Queue(CUSTOMERS_SEATS)  # A queue of size Customer Seats

    print("Opening barbershop...")
    
    # Create and start barber threads
    for i in range(BARBERS):
        barber = Barber(all_customers, i+1)  # Pass queue and barber ID
        barber.daemon = True  # Make daemon thread
        barber.start()   # Start barber thread
        barbers.append(barber)   # Add to barbers list
    
    print(f"Started {BARBERS} barbers")
    
    # Create customers
    customers = []
    for i in range(10):  # Create 10 customers
        print(f"---- Customer {i+1} ----")
        print(f"Current queue size: {all_customers.qsize()}")
        wait()  # Random wait between customer arrivals
        
        customer = Customer(all_customers)  # Create customer
        customer.start()  # Start customer thread (calls run method)
        customers.append(customer)
    
    # Wait for all customers to be processed
    print("Waiting for all customers to be served...")
    all_customers.join()    # Wait for all customers to be processed
    
    # Wait for customer threads to finish
    for customer in customers:
        customer.join()
    
    print(f"All customers served. Total earnings: €{Earnings}")
    
    # Close shop and wait for barbers to finish
    SHOP_OPEN = False
    EVENT.set()  # Wake up any sleeping barbers so they can exit
    
    for barber in barbers:
        barber.join(timeout=2)  # Wait for barbers to finish with timeout
    
    print("Barbershop closed.")
    print(f"Final earnings: €{Earnings}")