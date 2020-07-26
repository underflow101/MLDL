// Developer: Dongwon Paek
// Project: Machine Learning in C Language - Reinforcement Learning
// Subtitle: Q-Learning - Learning Optimized Way to Escape the Labyrinth (by robot itself)
// Since 2017.01.21
// Ended 2017.03.01
//*************************************************************************************

#define _CRT_SECURE_NO_WARNINGS

//*************************************************************************************
//Headers

#include <stdio.h>
#include <stdlib.h>

//*************************************************************************************
//Definitions

#define GENMAX 1000         //Repetition for learning
#define NODENO 15           //Q-value node
#define ALPHA 0.1           //Alpha value (Learning Coefficient)
#define GAMMA 0.9           //Gamma value (Discount Rate)
#define EPSILON 0.3         //Epsilon value (0 < e << 1 LOL (if you understand, you are ECE or EEE))
#define SEED 32767          //Seed for randomization

//*************************************************************************************
//Func Prototypes

int rand100();
int rand01();
double rand1();
void printqvalue(int qvalue[NODENO]);
int selectaction(int olds, int qvalue[NODENO]);
int updateq(int s, int qvalue[NODENO]);

//*************************************************************************************
//Main Func

int main() {
    int i, s, t;
    int qvalue[NODENO];

    srand(SEED);                            //Initialize randomization

    for(i = 0; i < NODENO; ++i) {           //Initialize Q-Value
        qvalue[i] = rand100();
    }
    printqvalue(qvalue);

    /* Important Part */
    for(i = 0; i < GENMAX; ++i) {           //Initialize machine learning process
        s = 0;
        for(t = 0; t < 3; ++t) {
            s = selectaction(s, qvalue);    //Select the act
            qvalue[s] = updateq(s, qvalue); //Update Q-value
        }
        printqvalue(qvalue);
    }

    return 0;
}

//*************************************************************************************
//Peripheral Func

int updateq(int s, int qvalue[NODENO]) {
    int qv, qmax;

    /* When at bottom most level */
    if(s > 6) {                 //When bottom most level
        if(s == 14) {           //Reward
            qv = qvalue[s] + ALPHA * (1000 - qvalue[s]);
        }
        /* When adding another nodes, delete this part from here... */
        else if(s == 11) {      //Increase the rewarding node to make machine learn easier
            qv = qvalue[s] + ALPHA * (500 - qvalue[s]);
        }
        /* ...delete until here */

        else {                  //No reward otherwise
            qv = qvalue[s];
        }
    }

    /* Otherwise (if != bottom most level) */
    else {
        if((qvalue[2 * s + 1]) > (qvalue[2 * s + 2])) {
            qmax = qvalue[2 * s + 1];
        } else {
            qv = qvalue[s] + ALPHA * (GAMMA * qmax - qvalue[s]);
        }
    }

    return qv;
}


int selectaction(int olds, int qvalue[NODENO]) {
    int s;

    /* Epsilon-Greedy Algorithm */
    if(rand1() < EPSILON) {
        if(rand01() == 0) {
            s = 2 * olds + 1;
        } else {
            s = 2 * olds + 2;
        }
    } else {
        if((qvalue[2 * olds + 1]) > (qvalue[2 * olds + 2])) {
            s = 2 * olds + 1;
        } else {
            s = 2 * olds + 2;
        }
    }

    return s;
}


void printqvalue(int qvalue[NODENO]) {
    int i;

    for(i = 1; i < NODENO; ++i) {
        printf("%d\t", qvalue[i]);
    }
    printf("\n");
}


double rand1() {
    return (double)rand() / RAND_MAX;
}


int rand01() {                                      //Returns random number between 0 ~ 1
    int rnd;

    while((rnd = rand()) == RAND_MAX);

    return (int)((double)rnd / RAND_MAX * 2);
}


int rand100() {                                     //Returns random number between 0 ~ 100
    int rnd;

    while((rnd = rand()) == RAND_MAX);

    return (int)((double)rnd / RAND_MAX * 101);
}
