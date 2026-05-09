const MILLISECONDS_PER_SECOND = 1000
const MILLISECONDS_PER_MINUTE = 60 * MILLISECONDS_PER_SECOND
const MILLISECONDS_PER_HOUR = 60 * MILLISECONDS_PER_MINUTE
const MILLISECONDS_PER_DAY = 24 * MILLISECONDS_PER_HOUR

export const humanizeMilliseconds = (milliseconds: number): string => {
    if (milliseconds < MILLISECONDS_PER_SECOND) {
        return `${milliseconds}ms`
    }

    if (milliseconds < MILLISECONDS_PER_MINUTE) {
        return `${Math.floor(milliseconds / MILLISECONDS_PER_SECOND)}s`
    }

    if (milliseconds < MILLISECONDS_PER_HOUR) {
        return `${Math.floor(milliseconds / MILLISECONDS_PER_MINUTE)}m`
    }

    if (milliseconds < MILLISECONDS_PER_DAY) {
        return `${Math.floor(milliseconds / MILLISECONDS_PER_HOUR)}h`
    }

    return `${Math.floor(milliseconds / MILLISECONDS_PER_DAY)}d`
}
