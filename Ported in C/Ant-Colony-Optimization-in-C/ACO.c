//Developer: Dongwon Paek
//Project: Machine Learning in C Language - Ant Colony Optimization
//Subtitle: ACO - Learning Optimized Way by Pheromone
//Since 2017.03.28.
//Ended 2017.04.14.
//*************************************************************************************

#define _CRT_SECURE_NO_WARNINGS

//*************************************************************************************
//Headers

#include <stdio.h>
#include <stdlib.h>

//*************************************************************************************
//Definitions

#define NOA 10          //Number of ants
#define ILIMIT 50       //Loop repetition
#define Q 3             //Pheromone update constant
#define RHO 0.8         //Rho - evaporation constant
#define STEP 10         //Number of steps for ways
#define EPSILON 0.15    //Epsilon value (0 < e << 1 LOL (if you understand, you are ECE or EEE))
#define SEED 32767      //As always

//*************************************************************************************
//Func Prototypes

void printp(double pheromone[2][STEP]);
void printmstep(int mstep[NOA][STEP]);
void walk(int cost[2][STEP], double pheromone[2][STEP], int mstep[NOA][STEP]);
void update(int cost[2][STEP], double pheromone[2][STEP], int mstep[NOA][STEP]);
double rand1();
int rand01();

//*************************************************************************************
//Main Func

int main() {
    int cost[2][STEP] = {{1,1,1,1,1,1,1,1,1,1}, {5,5,5,5,5,5,5,5,5,5}};
    double pheromone[2][STEP] = {0};
    int mstep[NOA][STEP];
    int i;

    srand(SEED);

    /* Optimization */
    for(i = 0; i < ILIMIT; ++i) {
        printf("%d: \n", i);
        printp(pheromone);
        walk(cost, pheromone, mstep);
        update(cost, pheromone, mstep);
    }

    return 0;
}

//*************************************************************************************
//Peripheral Func

void update(int cost[2][STEP], double pheromone[2][STEP], int mstep[NOA][STEP]) {
    int m, lm, i, j;
    double sum_lm = 0;

    /* Evaporation of Pheromone */
    for(i = 0; i < 2; ++i) {
        for(j = 0; j < STEP; ++j) {
            pheromone[i][j] *= RHO;
        }
    }

    /* Repetitive Pheromone Coating */
    for(m = 0; m < NOA; ++m) {
        lm = 0;                                               //Calculate distance moved
        for(i = 0; i < STEP; ++i) {
            lm += cost[mstep[m][i]][i] += Q * (1.0 / lm);
        }
        sum_lm += lm;
    }
    printf("%lf\n", sum_lm / NOA);
}


void walk(int cost[2][STEP], double pheromone[2][STEP], int mstep[NOA][STEP]) {
    int m, s;

    /* Epsilon-Greedy Algorithm */
    for(m = 0; m < NOA; ++m) {
        for(s = 0; s < STEP; ++s) {
            if((rand1() < EPSILON) || (abs(pheromone[0][s] - pheromone[1][s]) < 1e-9)) {
                mstep[m][s] = rand01();
            } else {
                if(pheromone[0][s] > pheromone[1][s]) {
                    mstep[m][s] = 0;
                } else {
                    mstep[m][s] = 1;
                }
            }
        }
    }

    printmstep(mstep);
}


void printmstep(int mstep[NOA][STEP]) {
    int i, j;

    printf("*mstep\n");
    for(i = 0; i < NOA; ++i) {
        for(j = 0; j < STEP; ++j) {
            printf("%d ", mstep[i][j]);
        }
        printf("\n");
    }
}


void printp(double pheromone[2][STEP]) {
    int i, j;

    for(i = 0; i < 2; ++i) {
        for(j = 0; j < STEP; ++j) {
            printf("%4.2lf ", pheromone[i][j]);
        }
        printf("\n");
    }
}


double rand1() {
    return (double)rand() / RAND_MAX;
}


int rand01() {
    int rnd;

    while((rnd = rand()) == RAND_MAX);

    return (int)((double)rnd / RAND_MAX * 2);
}
