#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include "common.h"

struct xtimer {
    struct timeval elapsed_time;
    struct timeval timestamp;
};

typedef struct xtimer   xtimer_t;

__BEGIN_C_DECLS

void timer_clear(xtimer_t *timer);
void timer_start(xtimer_t *timer);
void timer_stop(xtimer_t *timer);
double timer_elapsed_time(xtimer_t *timer);

__END_C_DECLS

#endif  /* TIMER_H */
