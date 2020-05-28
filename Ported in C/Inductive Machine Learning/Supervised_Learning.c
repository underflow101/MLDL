// Developer: Dongwon Paek
// Project: Machine Learning in C Language - Supervised Learning
// Subtitle: Inductive Machine Learning - Learning Stock
// Since 2017.07.11.
// Ended 2017.07.27.
//*************************************************************************************

#define _CRT_SECURE_NO_WARNINGS

//*************************************************************************************
// Headers

#include <stdio.h>
#include <stdlib.h>

//*************************************************************************************
// Definitions

#define SIZE 100            //Data's size
#define CNumber 10          //Corporation number
#define GENMAX 10000        //Max number of answer candidate generation
#define SEED 32767          //Seed for randomization

//*************************************************************************************
// Func Prototypes

void readdata(int data[SIZE][CNumber], int teacher[SIZE]);
int rand012();
int calcscore(int data[SIZE][CNumber], int teacher[SIZE], int answer[CNumber]);

//*************************************************************************************
// Main Func

int main() {
    int i, j, score = 0, bestscore = 0;
    int answer[CNumber];
    int data[SIZE][CNumber];
    int teacher[SIZE];
    int bestanswer[CNumber];

    srand(SEED);                            //Initialize randomization
    readdata(data, teacher);                //Read data for machine learning

    for(i = 0; i < GENMAX; ++i) {           //Generate & check answer candidate
        for(j = 0; j < CNumber; ++j) {      //Generate answer candidate
            answer[j] = rand012();
        }

        score = calcscore(data, teacher, answer);       //Check
        if(score > bestscore) {
            for(j = 0; j < CNumber; ++j) {
                bestanswer[j] = answer[j];
            }
            bestcore = score;

            for(j = 0; j < CNumber; ++j) {
                printf("%1d ", bestanswer[j]);
            }
            printf(": score = %d\n", bestscore);
        }
    }

    printf("\nThe best answer is...\n");
    for(j = 0; j < CNumber; ++j) {
        printf("%1d ", bestanswer[j]);
    }
    printf(": score = %d\n", bestscore);

    return 0;
}

//*************************************************************************************
// Peripheral Func

int calcscore(int data[SIZE][CNumber], int teacher[SIZE], int answer[CNumber]) {
    int score = 0, point, i, j;

    for(i = 0; i < SIZE; ++i) {     //Calculation for a match
        point = 0;
        for(j = 0; j < CNumber; ++j) {              //Point rises when it's a match, or a wild card
            if(answer[j] == 2) {
                ++point;
            } else if(answer[j] == data[i][j]) {
                ++point;
            }
        }

        if((point == CNumber) && (teacher[i] == 1)) {
            ++score;
        } else if((point != CNumber) && (teacher[i]) == 0) {
            ++score;
        }
    }

    return score;
}


void readdata(int data[SIZE][CNumber], int teacher[SIZE]) {
    int i, j;

    for(i = 0; i < SIZE; ++i) {
        for(j = 0; j < CNumber; ++j) {
            scanf("%d", &data[i][j]);
        }
        scanf("%d", &teacher[i]);
    }
}


void rand012() {
    int rnd;

    while((rnd = rand()) == RAND_MAX);      //Exclude maximum value of random number
    
    return (double)rnd / RAND_MAX * 3;      //Calculation for random number. Change data format to double.
}
