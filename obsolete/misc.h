#pragma once
#include <math.h>


typedef double DT_D;
typedef long DT_L;


DT_D get_dist(DT_D x1, DT_D y1, DT_D x2, DT_D y2) {
    return pow((((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2))), 0.5);
}


DT_L get_zero_idx(DT_D *x_arr, DT_L len_x) {
    DT_L i;

    for (i=0; i<len_x; ++i) {
        if (x_arr[i] == 0.0) {
            return i;
        }
    }
    return -1;
}


void fill_idw_wts_arr(DT_D *x_arr, DT_D *wts_arr, DT_D idw_exp, DT_L len_x) {
    DT_L i;

    for (i=0; i<len_x; ++i) {
        wts_arr[i] = 1.0 / pow(x_arr[i], idw_exp);
    }

    return;
}


void fill_mult_arr(DT_D *x, DT_D *y, DT_D *mult_arr, DT_L len_x) {
    DT_L i;

    for (i=0; i<len_x; ++i) {
        mult_arr[i] = x[i] * y[i];
    }

    return;
}


DT_D get_sum(DT_D *x, DT_L len_x) {
    DT_D _sum = 0;
    DT_L i = 0;

    for (i=0; i<len_x; ++i) {
        _sum += x[i];
    }

    return _sum;
}