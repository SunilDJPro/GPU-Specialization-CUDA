#include "assignment.h"

void executeTicketingSystemParticipation() {
	// Get a ticket number from the ticket machine
	ticketNumber = ++ticketMachineNumber;
	
	// For debugging purposes you might want to output the thread's ticketNumber
	std::cout << "Current thread ticket number: " << ticketNumber << "\n";
	
	std::string outputFileName = "output-" + currentPartId + ".txt";
	// NOTE: Do not remove this output to file statement as it is used to grade assignment,
	// so it should be called by each thread
	std::ofstream outputFile;
	outputFile.open (outputFileName, std::ofstream::app);
	outputFile << "C++11: Thread retrieved ticket number: " << ticketNumber << " started.\n";
	
	// wait until your ticket number has been called output your ticket number and the current time
	std::unique_lock<std::mutex> lock(ticketMutex);
	ticketCondition.wait(lock, [](){ return currentTicketNumber == ticketNumber; });

	// NOTE: Do not remove this output to file statement as it is used to grade assignment,
	// so it should be called by each thread
	outputFile << "C++11: Thread with ticket number: " << ticketNumber << " completed.\n";
	outputFile.close();
	
	// Increment currentTicketNumber variable to allow other threads to do their job
	currentTicketNumber++;
	ticketCondition.notify_all();
}

int runSimulation() {
	int result = 0;
	std::string userFromFile = getUsernameFromUserFile();
	if(USERNAME == currentUser && USERNAME == userFromFile){
		std::cout << "Simple user verification completed successfully.\n";
		
		// Initialize ticket system
		ticketMachineNumber = -1;  // Will start from 0 when first incremented
		currentTicketNumber = 0;   // First thread to execute has ticket 0
		
		std::thread threads[currentNumThreads];
		
		for(int threadIndex = 0; threadIndex < currentNumThreads; ++threadIndex){
			// This is where you will start threads that will participate in a ticketing system
			// have the thread run the executeTicketingSystemParticipation function
			threads[threadIndex] = std::thread(executeTicketingSystemParticipation);
		}
		
		// Let threads get their tickets first
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		
		// Start the ticketing system
		manageTicketingSystem();
		
		// Wait for all threads to complete
		for(int threadIndex = 0; threadIndex < currentNumThreads; ++threadIndex){
			threads[threadIndex].join();
		}

	} else {
		std::cout << "Simple user verification completed failed, code will not be executed.\n";
	}
	return result;
}

// Utility function for retrieving a user name from the .user file
std::string getUsernameFromUserFile(){
	std::string line;
	std::ifstream userFile (".user");
	if (userFile.is_open())
	{
		std::getline (userFile,line);
		userFile.close();
	}
	std::cout << "user from .user file: " << line << "\n";
	return line;
}

// Centralized logic for managing the ticketing machine that is the basis for threads executing work in the order that
//	they "pulled" the ticket from the machine
int manageTicketingSystem() {
	std::string outputFileName = "output-" + currentPartId + ".txt";
	std::ofstream outputFile;
	outputFile.open (outputFileName, std::ofstream::app);
	outputFile << "C++11: Signaling threads to do work.\n";
	
	// Notify all threads that they can start checking their tickets
	ticketCondition.notify_all();
	
	// Wait for all threads to complete
	// We wait until currentTicketNumber reaches the total number of threads
	std::unique_lock<std::mutex> lock(ticketMutex);
	ticketCondition.wait(lock, [](){ return currentTicketNumber >= currentNumThreads; });
	
	outputFile << "C++11: All threads completed.\n";
	outputFile.close();
	return 0;
}

int main(int argc, char *argv[]) {
	int numThreads = 1;
	std::string user = "coder";
	std::string partId = "test";
	std::cout << "Starting assignment main function\n";
	
	if (argc > 3) {
		std::cout << "Parsing command line arguments\n";
		numThreads = atoi(argv[1]);
		user = argv[2];
		partId = argv[3];
	}
	
	currentNumThreads = numThreads;
	currentUser = user;
	currentPartId = partId;
	
	// Note this implementation is based on a single code file and use of static and shared variables,
	// other implementations could be designed and are encouraged
	
	runSimulation();
}