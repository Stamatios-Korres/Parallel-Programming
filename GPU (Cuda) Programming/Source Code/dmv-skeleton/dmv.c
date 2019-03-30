/*
 *  dmv.c -- Dense matrix and vector operations for the CPU.
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "dmv.h"

void dmv_serial(const value_t *const *a, const value_t *x, value_t *y,
                size_t n)
{
    size_t  i, j;
    for (i = 0; i < n; ++i) {
        register value_t    _yi = 0;
        for (j = 0; j < n; ++j) {
            _yi += a[i][j]*x[j];
        }

        y[i] = _yi;
    }
}

void dmv_omp(const value_t *const *a, const value_t *x, value_t *y, size_t n)
{
    size_t  i, j;
#pragma omp parallel for private(i,j)
    for (i = 0; i < n; ++i) {
        register value_t    _yi = 0;
        for (j = 0; j < n; ++j) {
            _yi += a[i][j]*x[j];
        }

        y[i] = _yi;
    }
}

int vec_equals(const value_t *v1, const value_t *v2, size_t n, value_t eps)
{
    size_t  i;
    for (i = 0; i < n; ++i) {
        if (FABS(v1[i] - v2[i]) > eps)
            return i+1;
    }

    return 0;
}

void mat_transpose(value_t **a, size_t n)
{
    size_t  i, j;
    for (i = 0; i < n; ++i) {
        for (j = i+1; j < n; ++j) {
            value_t tmp = a[i][j];
            a[i][j] = a[j][i];
            a[j][i] = tmp;
        }
    }
}

void mat_init_rand(value_t **a, size_t n, value_t max)
{
    size_t  i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            a[i][j] = 2*(((value_t) drand48()) - MAKE_VALUE_CONSTANT(.5))*max;
        }
    }
}

void vec_init(value_t *v, size_t n, value_t val)
{
    size_t  i;
    for (i = 0; i < n; ++i) {
        v[i] = val;
    }
}

void vec_init_rand(value_t *v, size_t n, value_t max)
{
    size_t  i;
    for (i = 0; i < n; ++i) {
        v[i] = 2*(((value_t) drand48()) - MAKE_VALUE_CONSTANT(.5))*max;
    }
}

void vec_print(const value_t *v, size_t n)
{
    size_t  i;
    for (i = 0; i < n; ++i)
        printf("%" VALUE_FORMAT "\n", v[i]);
}
