/*
 *  common.h -- Basic definitions and declarations.
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#ifndef __COMMON_H
#define __COMMON_H

#include <stddef.h>
#define USEC_PER_SEC    1000000L

#undef  __BEGIN_C_DECLS
#undef  __END_C_DECLS

#if defined(__cplusplus) || defined(__CUDACC__)
#   define __BEGIN_C_DECLS  extern "C" {
#   define __END_C_DECLS    }
#else
#   define __BEGIN_C_DECLS
#   define __END_C_DECLS
#endif  /* __cplusplus || __CUDACC__ */

#endif  /* __COMMON_H */
