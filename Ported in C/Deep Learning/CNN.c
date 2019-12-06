//Developer: Dongwon Paek
//Project: DEEP Learning in C Language - Convolutional Neural Network
//Subtitle: Once And For All, CNN in C
//Since 2017.09.19.
//Ended 2018.01.08.
//*************************************************************************************

#define _CRT_SECURE_NO_WARNINGS

//*************************************************************************************
//Headers

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//*************************************************************************************
//Definitions

#define INPUTSIZE   11
#define FILTERSIZE  3
#define FILTERNO    2
#define POOLSIZE    3
#define POOLOUTSIZE 3
#define MAXINPUTNO  100         //Number of learning data
#define SEED        65535       //Random seed
#define LIMIT       0.001       //Limitation for error
#define BIGNUM      100         //Default number for error
#define HIDDENNO    3           //Number of middle layer cell
#define ALPHA       10          //Learning coefficient

//*************************************************************************************
//Func Prototypes

/* Convolution Product Calculation */
void conv(double filter[FILTERSIZE][FILTERSIZE], double e[][INPUTSIZE], double convout[][INPUTSIZE]);

/* Apply Filter */
double calcconv(double filter[][FILTERSIZE], double e[][INPUTSIZE], int i, int j);

/* Pooling Calculation */
void pool(double convout[][INPUTSIZE], double poolout[][POOLOUTSIZE]);

/* Maximum Number Pooling */
double maxpooling(double convout[][INPUTSIZE], int i, int j);

/* Read Data */
int getdata(double e[][INPUTSIZE], int r[]);

/* Show Data */
void showdata(double e[][INPUTSIZE][INPUTSIZE], int t[], int n_of_e);

/* Initialize Filter */
void initfilter(double filter[FILTERNO][FILTERSIZE][FILTERSIZE]);

/* Create Random Number */
double drnd();

/* Transfer Function == Sigmoid Function */
double f(double u);

/* Initialize Middle Layer Weight */
void initwh(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]);

/* Initialize Output Layer Weight */
void initwo(double wo[HIDDENNO + 1]);

/* Forward Direction Calculation */
double forward(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double wo[HIDDENNO + 1], double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]);

/* Adjust Output Layer Weight */
void olearn(double wo[HIDDENNO + 1], double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o);

/* Adjust Middle Layer Weight */
void hlearn(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double wo[HIDDENNO + 1], double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o);

/* Print Out Results */
void print(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double wo[HIDDENNO + 1]);

//*************************************************************************************
//Main Func

int main(int argc, char *argv) {
    double filter[FILTERNO][FILTERSIZE][FILTERSIZE];                        //Filter
    double e[MAXINPUTNO][INPUTSIZE][INPUTSIZE];                             //Input data
    int t[MAXINPUTNO];                                                      //Supervisor data
    double convout[INPUTSIZE][INPUTSIZE] = {0};                             //Convolution product output
    double poolout[POOLOUTSIZE][POOLOUTSIZE];                               //Output data
    int i, j, m, n;                                                         //i, j, m, n... you know what it means
    int n_of_e;                                                             //Number of learning data
    double err = BIGNUM;                                                    //Error evaluation
    int cnt = 0;                                                            //Counter
    double ef[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1];                    //Input data to Total bond layer
    double o;                                                               //Final output
    double hi[HIDDENNO + 1];                                                //Middle layer output
    double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1];          //Middle layer weight
    double wo[HIDDENNO + 1];                                                //Output layer weight

    srand(SEED);
    initfilter(filter);
    initwh(wh);
    initwo(wo);

    n_of_e = getdata(e, t);
    showdata(e, t, n_of_e);

    /* Deep Learning Starts */
    while(err > LIMIT) {
        err = 0.0;
        for(i = 0; i < n_of_e; ++i) {
            for(j = 0; j < FILTERNO; ++j) {
                conv(filter[j], e[i], convout);
                pool(convout, poolout);

                for(m = 0; m < POOLOUTSIZE; ++m) {
                    for(n = 0; n < POOLOUTSIZE; ++n) {
                        ef[j * POOLOUTSIZE * POOLOUTSIZE + POOLOUTSIZE * m + n] = poolout[m][n];
                    }
                }

                ef[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] = t[i];            //Supervisor data
            }

            o = forward(wh, wo, hi, ef);                //Forward direction calculation
            olearn(wo, hi, ef, o);                      //Adjust output layer weight
            hlearn(wh, wo, hi, ef, o);                  //Adjust middle layer weight
            err += (o - t[i]) * (o - t[i]);             //Error calculation
        }
        ++cnt;
        fprintf(stderr, "%d\t%lf\n", cnt, err);
    }
    /* Deep Learning Ends */

    printf("\n*******Results*******\n");
    printf("Weights\n");
    print(wh, wo);

    printf("Network output\n");
    printf("#\tteacher\toutput\n");

    for(i = 0; i < n_of_e; ++i) {
        printf("%d\t%d\t", i, t[i]);

        for(j = 0; j < FILTERNO; ++j) {
            conv(filter[j], e[i], convout);
            pool(convout, poolout);

            for(m = 0; m < POOLOUTSIZE; ++m) {
                for(n = 0; n <POOLOUTSIZE; ++n) {
                    ef[j * POOLOUTSIZE * POOLOUTSIZE + POOLOUTSIZE * m + n] = poolout[m][n];
                }
            }
            ef[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] = t[i];             //Supervisor data
        }

        o = forward(wh, wo, hi, ef);
        printf("%lf\n", o);
    }

    return 0;
}

//*************************************************************************************
//Peripheral Func

void print(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double wo[HIDDENNO + 1]) {
    int i, j;

    for(i = 0; i < HIDDENNO; ++i) {
        for(j = 0; j < POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1; ++j) {
            printf("%lf ", wh[i][j]);
        }
    }
    printf("\n");

    for(i = 0; i < HIDDENNO + 1; ++i) {
        printf("%lf ", wo[i]);
    }
    printf("\n");
}


void hlearn(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double wo[HIDDENNO + 1], double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o) {
    int i, j;
    double dj;

    for(j = 0; j < HIDDENNO; ++j) {
        dj = hi[j] * (1 - hi[j]) * wo[j] * (e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] - o) * o * (1 - o);     //Process 1st weight

        for(i = 0; i < POOLOUTSIZE * POOLOUTSIZE * FILTERNO; ++i) {
            wh[j][i] += ALPHA * e[i] * dj;
        }

        wh[j][i] += ALPHA * (-1.0) * dj;                //Learn threshold value
    }
}


void olearn(double wo[HIDDENNO + 1], double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o) {
    int i;
    double d;

    d = (e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] - o) * o * (1 - o);          //Error calculation
    for(i = 0; i < HIDDENNO; ++i) {
        wo[i] += ALPHA * hi[i] * d;
    }
    wo[i] += ALPHA * (-1.0) * d;                                            //Learn threshold value
}


double forward(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double wo[HIDDENNO + 1], double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]) {
    int i, j;
    double u;       //Calculate sum of weights
    double o;       //Output calculation

    /* Calculate hi */
    for(i = 0; i < HIDDENNO; ++i) {
        u = 0;
        for(j = 0; j < POOLOUTSIZE * POOLOUTSIZE * FILTERNO; ++j) {
            u += e[j] * wh[i][j];
        }
        u -=wh[i][j];        //Process threshold value
        hi[i] = f(u);
    }

    o = 0;
    for(i = 0; i < HIDDENNO; ++i) {
        o += hi[i] * wo[i];
    }
    o -= wo[i];             //Process threshold value

    return f(o);
}


void initwh(double wh[][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]) {
    int i, j;

    for(i = 0; i < HIDDENNO; ++i) {
        for(j = 0; j < HIDDENNO; ++j) {
            wh[i][j] = drnd();
        }
    }
}


void initwo(double wo[]) {
    int i;

    for(i = 0; i < HIDDENNO + 1; ++i) {
        wo[i] = drnd();
    }
}


void initfilter(double filter[FILTERNO][FILTERSIZE][FILTERSIZE]) {
    int i, j, k;

    for(i = 0; i < FILTERNO; ++i) {
        for(j = 0; j <FILTERSIZE; ++j) {
            for(k = 0; k < FILTERSIZE; ++k) {
                filter[i][j][k] = drnd();
            }
        }
    }
}


double drnd() {
    double rndno;

    while((rndno = (double)rand() / RAND_MAX) == 1.0);

    rndno *= 2 - 1;
    
    return rndno;
}


void pool(double convout[][INPUTSIZE], double poolout[][POOLOUTSIZE]) {
    int i, j;

    for(i = 0; i < POOLOUTSIZE; ++i) {
        for(j = 0; j < POOLOUTSIZE; ++j) {
            poolout[i][j] = maxpooling(convout, i, j);
        }
    }
}


double maxpooling(double convout[][INPUTSIZE], int i, int j) {
    int m, n;
    double max;
    int halfpool = POOLSIZE / 2;

    max = convout[i * POOLOUTSIZE + 1 + halfpool][j * POOLOUTSIZE + 1 + halfpool];

    for(m = POOLOUTSIZE * i + 1; m <= POOLOUTSIZE * i + 1 + (POOLSIZE - halfpool); ++m) {
        for(n = POOLOUTSIZE * j + 1; n <= POOLOUTSIZE * j + 1 + (POOLSIZE - halfpool); ++n) {
            if(max < convout[m][n]) {
                max = convout[m][n];
            }
        }
    }

    return max;
}


void showdata(double e[][INPUTSIZE][INPUTSIZE], int t[], int n_of_e) {
    int i = 0, j = 0, k = 0;

    for(i = 0; i < n_of_e; ++i) {
        printf("N=%d category=$d\n", i, t[i]);
        for(j = 0; j < INPUTSIZE; ++j) {
            for(k = 0; k < INPUTSIZE; ++k) {
                printf("%.3lf ", e[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


int getdata(double e[][INPUTSIZE], int t[]) {
    int i = 0, j = 0, k = 0;

    while(scanf("%d", &t[i]) != EOF) {
        ++k;
        if(j >= INPUTSIZE) {
            k = 0;
            ++k;
            if(j >= INPUTSIZE) {
                break;
            }
        }
        j = 0; k = 0;
        ++i;
    }
    return i;
}


void conv(double filter[FILTERSIZE][FILTERSIZE], double e[][INPUTSIZE], double convout[][INPUTSIZE]) {
    int i = 0, j = 0;
    int startpoint = FILTERSIZE / 2;

    for(i = startpoint; i < INPUTSIZE - startpoint; ++i) {
        for(j = startpoint; j < INPUTSIZE - startpoint; ++j) {
            convout[i][j] = calcconv(filter, e, i, j);
        }
    }
}


double calcconv(double filter[][FILTERSIZE], double e[][INPUTSIZE], int i, int j) {
    int m, n;
    double sum = 0;

    for(m = 0; m < FILTERSIZE; ++m) {
        for(n = 0; n < FILTERSIZE; ++n) {
            sum += e[i - FILTERSIZE / 2 + m][j - FILTERSIZE / 2 + n] * filter[m][n];
        }
    }

    return sum;
}


double f(double u) {
    return 1.0 / (1.0 + exp(-u));
}
