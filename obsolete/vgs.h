#pragma once
#include <math.h>


typedef double DT_D;


DT_D rng_vg(DT_D h, DT_D r, DT_D s) {
    return h;
}


DT_D nug_vg(DT_D h, DT_D r, DT_D s) {
    return s;
}


DT_D sph_vg(DT_D h, DT_D r, DT_D s) {
    DT_D a_1, b_1;

    if (h >= r) {
        return s;
    }
    else {
        a_1 = (1.5 * h) / r;
        b_1 = (h*h*h) / (2 * (r*r*r));
        return (s * (a_1 - b_1));
    }
}


DT_D exp_vg(DT_D h, DT_D r, DT_D s) {
    return (s * (1 - exp(-3 * h / r)));
}


DT_D lin_vg(DT_D h, DT_D r, DT_D s) {
    if (h > r) {
        return s;
    }
    else {
        return s * (h / r);
    }
}


DT_D gau_vg(DT_D h, DT_D r, DT_D s) {
    return (s * (1 - exp(-3 * (((h*h) / (r*r))))));
}


DT_D pow_vg(DT_D h, DT_D r, DT_D s) {
    return (s * (h**r));
}


DT_D hol_vg(DT_D h, DT_D r, DT_D s) {
    DT_D a_1, hol_vg = 0.0;

    if (h != 0) {
        a_1 = (M_PI * h) / r;
        hol_vg = (s * (1 - (sin(a_1)/a_1)));
    }

    return hol_vg;
}
