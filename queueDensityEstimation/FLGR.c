/*****************************************************************
    FrugalLight C++ Implementation (Group, Relative)
    Version 1.0
    Author: sachin-iitd@github
*****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RL              2   /* 1 or 2 */
#define NUM_APPROACH    4
#define ABSOLUTE_PHASE  0   /* 0 or 1 */
#define VERBOSE         0   /* 0 or 1*/

// FL-G
typedef unsigned long long dtyp;
#define QUANTIZE_LEVELS 21
dtyp lutFLG[QUANTIZE_LEVELS+1];

// FL-R
#define ALPHA           0.2
#define MAX_GREEN       90
#define MIN_GREEN       5
#define TIME_INTERVAL   5
int phase_time = 0;
int action = 0;

void initFLG(void)
{
    // Eventually load model from file
    for (int i=0 ; i<=QUANTIZE_LEVELS ; i++)
    {
        lutFLG[i] = (((dtyp)rand())<<32) + rand();
#if VERBOSE
        printf("lut[%2d] = 0x%016llx\n", i, lutFLG[i]);
#endif
    }
}

int GetNextPhaseFLG(float* density, int cur_phase)
{
    float one_bin = 1.0/QUANTIZE_LEVELS;
    int green = (int)(density[cur_phase]/one_bin + 0.5);
    float redf = 0;
    for (int i=0 ; i<NUM_APPROACH ; ++i)
    {
        if (i != cur_phase)
            redf += density[i]/one_bin;
    }
    int red = (int)(redf + 0.5);

    if (green > QUANTIZE_LEVELS)
        green = QUANTIZE_LEVELS;
    if (red > 3*QUANTIZE_LEVELS)
        red = 3*QUANTIZE_LEVELS;

    int action = (lutFLG[green] & (((dtyp)1)<<red)) != 0 ;
#if VERBOSE
    printf("Green = %d, Red = %d, Act = %d\n", green, red, action);
#endif
#if ABSOLUTE_PHASE
    return (action + cur_phase) % NUM_APPROACH;
#else
    return action;
#endif
}

int GetNextPhaseFLR(float* density, int cur_phase)
{
    // if(cur_phase==1){
    //     return 4;
    // }
    int green = density[cur_phase];
    float all = 0;
    for (int i=0 ; i<NUM_APPROACH ; ++i)
        all += density[i];
    if (all < 1e-6)  // Almost 0
        return 0;
    float relative_density = green / all;
    if (relative_density >= ALPHA || phase_time <= MIN_GREEN)
        return 0;


    int action = (MAX_GREEN <= 0) ? (rand()%100 > relative_density*100) : (phase_time/MAX_GREEN > relative_density);
    if (action)
        phase_time = 0;
    else
        phase_time += TIME_INTERVAL;
#if VERBOSE
    printf("Rel Density = %f, PhaseTime = %d, Act = %d\n", relative_density, phase_time, action);
#endif
#if ABSOLUTE_PHASE
    return (action + cur_phase) % NUM_APPROACH;
#else
    return action;
#endif
}

int main(void)
{
    float density[NUM_APPROACH] = {0.5, 0.33, 0.33, 0.33};
    int cur_phase = 1;

    initFLG();
    printf("FL-G : %d\n", GetNextPhaseFLG(density, cur_phase));
    printf("FL-R : %d\n", GetNextPhaseFLR(density, cur_phase));
}