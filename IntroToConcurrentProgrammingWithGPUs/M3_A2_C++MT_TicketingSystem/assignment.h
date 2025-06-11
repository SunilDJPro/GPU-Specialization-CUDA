#include <thread>
#include <iostream>
#include <fstream>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>

#ifndef CPP_INITIAL_CODE_ASSIGNMENT_H
#define CPP_INITIAL_CODE_ASSIGNMENT_H

const std::string USERNAME = "coder";
static std::atomic<int> currentTicketNumber; // Atomic for thread safety
static std::atomic<int> ticketMachineNumber; // For assigning tickets to threads
static std::string currentPartId;
static std::string currentUser;
static int currentNumThreads;

// Synchronization primitives
static std::mutex ticketMutex;
static std::condition_variable ticketCondition;

// Thread-local storage for ticket numbers
thread_local int ticketNumber;

void executeTicketingSystemParticipation();
int runSimulation();
std::string getUsernameFromUserFile();
int manageTicketingSystem();

#endif //CPP_INITIAL_CODE_ASSIGNMENT_H